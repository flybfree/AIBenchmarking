"""Aggregation of per-request results into comparable summary statistics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable

from .client import RequestResult


def _stat(values: list[float], fn) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return fn(vals)


def _pct(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


# A TTFT counts as a "spike" when it exceeds both an absolute floor and this
# multiple of the group's median — i.e. an outlier well above typical latency
# (usually inference-server jitter, not steady-state behaviour).
_SPIKE_FACTOR = 3.0
_SPIKE_FLOOR_MS = 250.0


def _spike_rate(ttft_ms: list[float]) -> float | None:
    vals = [v for v in ttft_ms if v is not None]
    if len(vals) < 2:
        return None
    med = statistics.median(vals)
    thresh = max(_SPIKE_FLOOR_MS, _SPIKE_FACTOR * med)
    return sum(1 for v in vals if v > thresh) / len(vals)


@dataclass
class Aggregate:
    """Summary statistics over repeated runs of one (endpoint, group)."""

    endpoint: str
    model: str
    hardware: str
    group: str                 # task id or category
    n: int
    n_ok: int

    tokens_per_s_mean: float | None = None
    tokens_per_s_median: float | None = None
    tokens_per_s_std: float | None = None    # sample std dev (n>=2)
    tokens_per_s_n: int = 0                   # samples behind the tok/s stats
    ttft_ms_mean: float | None = None
    ttft_ms_p95: float | None = None
    ttft_spike_rate: float | None = None   # fraction of runs with TTFT >> median
    total_s_mean: float | None = None
    completion_tokens_mean: float | None = None
    tokens_estimated: bool = False
    reasoning: bool = False            # model emitted thinking tokens
    content_empty_n: int = 0           # runs that produced no visible answer
    tps_fallback_n: int = 0            # runs whose tok/s fell back to /total_s
    quality_mean: float | None = None  # Phase 2 reference score in [0, 1]
    quality_n: int = 0                 # runs with a quality score
    judge_mean: float | None = None    # Phase 3 LLM-judge score in [0, 1]
    judge_n: int = 0                   # runs with a judge score
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def aggregate(
    results: Iterable[RequestResult],
    group_key: str = "task",
) -> list[Aggregate]:
    """Group results by (endpoint, group_key) and compute summary stats.

    `group_key` is "task" (per prompt) or "category" — for category grouping,
    callers should set `RequestResult.task` to the category before aggregating,
    or use `aggregate_by_category`.
    """

    buckets: dict[tuple[str, str], list[RequestResult]] = {}
    for r in results:
        key = (r.endpoint, getattr(r, group_key))
        buckets.setdefault(key, []).append(r)

    out: list[Aggregate] = []
    for (endpoint, group), rs in buckets.items():
        ok = [r for r in rs if r.ok]
        sample = ok[0] if ok else rs[0]
        tps = [r.tokens_per_s for r in ok]
        ttft = [r.ttft_s * 1000 for r in ok if r.ttft_s is not None]
        quals = [r.quality for r in ok if getattr(r, "quality", None) is not None]
        judges = [r.judge_score for r in ok if getattr(r, "judge_score", None) is not None]
        agg = Aggregate(
            endpoint=endpoint,
            model=sample.model,
            hardware=sample.hardware,
            group=group,
            n=len(rs),
            n_ok=len(ok),
            tokens_per_s_mean=_stat(tps, statistics.mean),
            tokens_per_s_median=_stat(tps, statistics.median),
            tokens_per_s_std=(statistics.stdev([v for v in tps if v is not None])
                              if len([v for v in tps if v is not None]) >= 2 else None),
            tokens_per_s_n=len([v for v in tps if v is not None]),
            ttft_ms_mean=_stat(ttft, statistics.mean),
            ttft_ms_p95=_pct(ttft, 0.95),
            ttft_spike_rate=_spike_rate(ttft),
            total_s_mean=_stat([r.total_s for r in ok], statistics.mean),
            completion_tokens_mean=_stat(
                [float(r.completion_tokens) for r in ok if r.completion_tokens],
                statistics.mean,
            ),
            tokens_estimated=any(r.tokens_estimated for r in ok),
            reasoning=any(getattr(r, "reasoning", False) for r in ok),
            content_empty_n=sum(1 for r in ok if getattr(r, "content_empty", False)),
            tps_fallback_n=sum(1 for r in ok if getattr(r, "tps_from_total", False)),
            quality_mean=_stat(quals, statistics.mean),
            quality_n=len(quals),
            judge_mean=_stat(judges, statistics.mean),
            judge_n=len(judges),
            errors=[r.error for r in rs if r.error][:5],
        )
        out.append(agg)
    out.sort(key=lambda a: (a.group, -(a.tokens_per_s_mean or 0)))
    return out


def aggregate_by_category(results: list[RequestResult]) -> list[Aggregate]:
    """Aggregate rolled up to task category instead of individual task id."""
    return aggregate(results, group_key="category")
