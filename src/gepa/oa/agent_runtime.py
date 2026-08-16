"""Provider-neutral workspace-agent injection seam for agentic engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol


@dataclass(frozen=True)
class AgentRunRequest:
    continuation_id: str
    resume: bool
    prompt: str
    cwd: Path
    model: str
    reasoning_effort: str | None
    sandbox: Literal["read-only", "workspace-write"]
    timeout_seconds: float
    stop_requested: Callable[[], str | None] | None = None


@dataclass(frozen=True)
class AgentRunResult:
    text: str
    thread_id: str
    status: Literal["completed", "failed", "interrupted", "ambiguous"]
    usage: dict[str, int]
    cost_usd: float | None = None


class AgentRunner(Protocol):
    def run(self, request: AgentRunRequest) -> AgentRunResult: ...
