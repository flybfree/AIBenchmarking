"""Apply reference-based scoring to a set of results.

Scoring is a post-processing step: it runs over `RequestResult`s (fresh or
loaded from a saved run) and attaches quality scores, so past runs can be
scored without re-benchmarking. Tasks are resolved by id from the built-in
suites; results whose task has no checks are left unscored.
"""

from __future__ import annotations

from ..client import RequestResult
from ..tasks import ALL_SUITES, Task
from .base import get_scorer


def _tasks_by_id() -> dict[str, Task]:
    out: dict[str, Task] = {}
    for suite in ALL_SUITES.values():
        for t in suite.tasks:
            out[t.id] = t
    return out


def score_results(
    results: list[RequestResult],
    scorer_name: str = "metrics",
) -> int:
    """Score results in place. Returns the number of results scored."""
    scorer = get_scorer(scorer_name)
    tasks = _tasks_by_id()
    scored = 0
    for r in results:
        task = tasks.get(r.task)
        if task is None or not task.checks:
            continue
        s = scorer.score(task, r)
        if "quality" in s.scores:
            r.quality = s.scores["quality"]
            r.check_scores = {k: v for k, v in s.scores.items() if k != "quality"}
            r.score_note = s.notes
            scored += 1
    return scored
