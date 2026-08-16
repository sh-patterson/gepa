from __future__ import annotations

import json
from pathlib import Path

from gepa.oa.agent_runtime import AgentRunRequest, AgentRunResult
from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engines.autoresearch import AutoResearchEngine
from gepa.oa.engines.meta_harness import _run_proposer


class RecordingRunner:
    def __init__(self) -> None:
        self.requests: list[AgentRunRequest] = []

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.requests.append(request)
        return AgentRunResult(
            text="done",
            thread_id="runtime-thread-1",
            status="completed",
            usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            cost_usd=None,
        )


def test_autoresearch_uses_agent_runner_and_preserves_continuation(tmp_path: Path) -> None:
    runner = RecordingRunner()
    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            engine_config={
                "model": "gpt-5.6-luna",
                "effort": "high",
                "agent_runner": runner,
                "agent_timeout_seconds": 12,
            },
        )
    )
    budget = BudgetTracker(max_evals=10)

    first = engine._run_claude(
        work_dir=tmp_path,
        session_id="continuation-1",
        prompt="first",
        budget=budget,
        adapter_cost=0,
        resume=False,
        env={},
    )
    second = engine._run_claude(
        work_dir=tmp_path,
        session_id="continuation-1",
        prompt="second",
        budget=budget,
        adapter_cost=0,
        resume=True,
        env={},
    )

    assert first.returncode == second.returncode == 0
    assert [request.resume for request in runner.requests] == [False, True]
    assert {request.continuation_id for request in runner.requests} == {"continuation-1"}
    assert all(request.sandbox == "workspace-write" for request in runner.requests)
    assert all(request.timeout_seconds == 12 for request in runner.requests)


def test_meta_harness_uses_isolated_agent_runner_and_native_skill(tmp_path: Path) -> None:
    class WritingRunner(RecordingRunner):
        def run(self, request: AgentRunRequest) -> AgentRunResult:
            (request.cwd / "agents").mkdir(exist_ok=True)
            (request.cwd / "agents" / "candidate.txt").write_text("BLUE")
            (request.cwd / "state").mkdir(exist_ok=True)
            (request.cwd / "state" / "pending.json").write_text(
                json.dumps([{"name": "candidate", "file": "agents/candidate.txt"}])
            )
            return super().run(request)

    runner = WritingRunner()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "state").mkdir()
    log_dir = work_dir / "sessions"
    exit_code, cost, thread_id = _run_proposer(
        work_dir=work_dir,
        iteration=1,
        model="gpt-5.6-luna",
        effort="high",
        max_candidates=3,
        max_budget_usd=None,
        pending_path=work_dir / "state" / "pending.json",
        log_dir=log_dir,
        sandbox=True,
        agent_runner=runner,
        agent_timeout_seconds=15,
    )

    assert exit_code == 0
    assert cost == 0
    assert thread_id == "runtime-thread-1"
    assert len(runner.requests) == 1
    assert runner.requests[0].resume is False
    assert ".agents/skills/" in runner.requests[0].prompt
    meta = json.loads((log_dir / "iter1_meta.json").read_text())
    assert meta["cost_status"] == "unknown"


def test_autoresearch_native_runner_exposes_resettable_progress_watchdog(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            engine_config={
                "agent_runner": runner,
                "agent_timeout_seconds": 60,
                "max_no_eval_seconds": 0.01,
            },
        )
    )

    engine._run_claude(
        work_dir=tmp_path,
        session_id="continuation-watchdog",
        prompt="work",
        budget=BudgetTracker(max_evals=10),
        adapter_cost=0,
        resume=False,
        env={},
    )

    request = runner.requests[0]
    assert request.timeout_seconds == 60
    assert request.stop_requested is not None


def test_autoresearch_native_watchdog_resets_after_eval_progress(monkeypatch, tmp_path: Path) -> None:
    now = [0.0]
    monkeypatch.setattr("gepa.oa.engines.autoresearch.time.monotonic", lambda: now[0])
    budget = BudgetTracker(max_evals=10)

    class WatchingRunner(RecordingRunner):
        def run(self, request: AgentRunRequest) -> AgentRunResult:
            assert request.stop_requested is not None
            now[0] = 0.009
            assert request.stop_requested() is None
            budget.record(1.0)
            now[0] = 0.015
            assert request.stop_requested() is None
            now[0] = 0.026
            assert request.stop_requested() is not None
            return super().run(request)

    engine = AutoResearchEngine(
        OptimizeAnythingConfig(
            engine="autoresearch",
            engine_config={
                "agent_runner": WatchingRunner(),
                "agent_timeout_seconds": 60,
                "max_no_eval_seconds": 0.01,
            },
        )
    )
    engine._run_claude(
        work_dir=tmp_path,
        session_id="continuation-watchdog-reset",
        prompt="work",
        budget=budget,
        adapter_cost=0,
        resume=False,
        env={},
    )
