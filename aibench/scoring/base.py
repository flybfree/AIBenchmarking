"""Pluggable scoring interface.

A Scorer takes a completed `RequestResult` (and the `Task` it answered) and
returns a dict of named scores in [0, 1] plus optional notes. Phase 1 ships only
the performance-oriented pass-through; Phase 2 (reference metrics) and Phase 3
(LLM-judge) implement this same protocol and are registered in `REGISTRY`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..client import RequestResult
from ..tasks import Task


@dataclass
class Score:
    scores: dict[str, float] = field(default_factory=dict)  # name -> [0,1]
    notes: str = ""


class Scorer(Protocol):
    name: str

    def score(self, task: Task, result: RequestResult) -> Score: ...


class NullScorer:
    """Phase 1: no quality scoring; performance metrics carry the comparison."""

    name = "none"

    def score(self, task: Task, result: RequestResult) -> Score:
        return Score()


# Phase 2/3 scorers register themselves here (e.g. "metrics", "judge").
REGISTRY: dict[str, Scorer] = {"none": NullScorer()}


def get_scorer(name: str) -> Scorer:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown scorer '{name}'. Available: {', '.join(REGISTRY)}"
        )
    return REGISTRY[name]
