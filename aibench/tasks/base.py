"""Task definitions.

A `Task` is a single prompt (plus generation params and optional tool schema)
tagged with a category. A `TaskSuite` groups related tasks. Phase 1 uses only
the prompt + params for performance measurement; the extra fields (`reference`,
`expects_tool`, `rubric`) are carried through now so Phase 2/3 scorers can use
them without changing the task format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    id: str
    category: str                          # creative_writing | code_generation | ...
    prompt: str
    difficulty: str = "medium"             # easy | medium | hard
    system: str | None = None
    params: dict[str, Any] = field(default_factory=dict)   # max_tokens, temperature
    tools: list[dict[str, Any]] = field(default_factory=list)

    # Multi-turn agent tasks: tool_impls maps a tool name to a callable
    # (args_dict -> result string) the harness runs between turns. When
    # tool_impls is set the runner drives an agent loop for up to max_turns.
    tool_impls: dict[str, Any] = field(default_factory=dict)
    max_turns: int = 1

    # Phase 2 reference-based scoring: a list of check specs (see
    # aibench/scoring/checks.py). Each check returns a sub-score in [0, 1];
    # the task's quality score is their weighted mean.
    checks: list[dict[str, Any]] = field(default_factory=list)

    # Reserved / metadata hooks.
    reference: str | None = None           # source text (kept for context)
    expects_tool: str | None = None        # expected tool name
    rubric: str | None = None              # Phase 3: LLM-judge rubric

    @property
    def is_agent(self) -> bool:
        return bool(self.tool_impls)

    def messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.prompt})
        return msgs


@dataclass
class TaskSuite:
    name: str
    description: str
    tasks: list[Task]
