"""Reference-based scorer (Phase 2).

Combines a task's `checks` into a single quality score in [0, 1] (the weighted
mean of the checks' sub-scores). Registers itself as the "metrics" scorer.
"""

from __future__ import annotations

from ..client import RequestResult
from ..tasks import Task
from .base import Score, REGISTRY
from .checks import run_check


class ReferenceScorer:
    name = "metrics"

    def score(self, task: Task, result: RequestResult) -> Score:
        if not task.checks:
            return Score(notes="no checks defined for this task")
        if not result.ok:
            return Score(scores={"quality": 0.0}, notes="request failed")

        scores: dict[str, float] = {}
        notes: list[str] = []
        wsum = 0.0
        acc = 0.0
        for i, spec in enumerate(task.checks):
            w = float(spec.get("weight", 1.0))
            s, note = run_check(task, result, spec)
            label = spec.get("type", f"check{i}")
            scores[label] = round(s, 3)
            notes.append(f"{label}: {note}")
            acc += w * s
            wsum += w
        scores["quality"] = round(acc / wsum, 3) if wsum else 0.0
        return Score(scores=scores, notes=" | ".join(notes))


REGISTRY["metrics"] = ReferenceScorer()
