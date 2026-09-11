"""Aggregate multiple saved runs into one cross-model leaderboard.

Each run reports one config's endpoints; over many runs you accumulate results
for many models. This pools them by (model @ hardware) — the deployable unit —
so you can compare every model you've benchmarked in a single view. Quality is a
model property and pools cleanly across runs; throughput is model x hardware x
run-conditions, so it's pooled too but is noisier (different concurrency/load
across runs) — shown with a sample count so you can judge how solid it is.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .client import RequestResult


@dataclass
class Cell:
    objective: float | None = None
    judge: float | None = None
    quality: float | None = None       # effective: judge if judged, else objective
    basis: str = ""                    # "judged" | "checks" | "failed"
    tps: float | None = None
    tps_std: float | None = None
    n: int = 0                         # measured samples pooled
    quality_excluded_runs: int = 0     # runs dropped from quality (truncated/broken)
    failed: bool = False               # category attempted but every request errored
    failed_n: int = 0                  # failed requests behind a failed cell


@dataclass
class ModelRow:
    model: str
    hardware: str
    cats: dict[str, Cell] = field(default_factory=dict)
    runs: set[str] = field(default_factory=set)
    samples: int = 0
    offloaded_runs: int = 0            # runs excluded from throughput pooling
    overall_quality: float | None = None  # mean over attempted cats; failed count as 0
    cats_attempted: int = 0            # categories this unit was actually run on
    cats_ok: int = 0                   # attempted categories with any successful result


@dataclass
class LoadedRuns:
    results: list[RequestResult]
    run_meta: list[dict]               # {label, file, started, models}


def load_runs(paths: list[str | Path]) -> LoadedRuns:
    """Load result JSONs, tagging each result with its source run label."""
    results: list[RequestResult] = []
    meta: list[dict] = []
    for p in paths:
        p = Path(p)
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cfg = payload.get("config", {})
        label = cfg.get("label") or p.stem
        run_id = f"{label} ({p.stem})"
        for rd in payload.get("results", []):
            try:
                r = RequestResult(**rd)
            except TypeError:
                # Tolerate schema drift: keep only known fields.
                known = RequestResult.__dataclass_fields__
                r = RequestResult(**{k: v for k, v in rd.items() if k in known})
            setattr(r, "run_label", run_id)
            results.append(r)
        meta.append({
            "label": label,
            "file": p.name,
            "started": payload.get("started_at", "")[:19].replace("T", " "),
            "models": sorted({e.get("model", "") for e in cfg.get("endpoints", [])}),
        })
    return LoadedRuns(results, meta)


# A run's cell is dropped from quality pooling when more than this fraction of
# its responses were "broken" (empty + not a clean stop/length + tiny output) —
# the signature of a truncation/format bug, distinct from a genuine overflow
# (finish="length" at the full token budget), which stays counted as a real
# failure.
BROKEN_RUN_FRAC = 0.34


def _is_broken(r: RequestResult) -> bool:
    return (
        r.ok
        and not (r.text or "").strip()
        and not r.tool_calls
        and r.finish_reason not in ("stop", "length")
        and (r.completion_tokens or 0) < 20
    )


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _offloaded_runs_for_unit(unit_results: list[RequestResult]) -> set[str]:
    """Which of this (model @ hardware)'s runs look CPU-offloaded — flat, low
    throughput — so they can be excluded from throughput pooling. Reuses the
    single-endpoint offload thresholds."""
    from .diagnose import OFFLOAD_TPS_FLOOR, OFFLOAD_FLAT_CV
    by_run: dict[str, list[float]] = {}
    for r in unit_results:
        if r.tokens_per_s:
            by_run.setdefault(getattr(r, "run_label", "?"), []).append(r.tokens_per_s)
    bad = set()
    for run, tps in by_run.items():
        if len(tps) < 3:
            continue
        mean = statistics.mean(tps)
        if mean and statistics.median(tps) < OFFLOAD_TPS_FLOOR \
                and statistics.stdev(tps) / mean < OFFLOAD_FLAT_CV:
            bad.add(run)
    return bad


def aggregate_by_model(results: list[RequestResult]) -> tuple[list[ModelRow], list[str]]:
    """Pool results into (model @ hardware) rows with per-category cells.

    Quality pools across all runs (offload doesn't change correctness), but
    throughput excludes any run in which this unit was CPU-offloaded, so one bad
    run doesn't drag the pooled tok/s."""
    categories = sorted({r.category for r in results if r.category})
    # Track ALL results (ok or errored) so we can tell a category that was
    # attempted-but-wholly-failed (a real capability/serving failure, counts
    # against the model) from one that was simply never run (no data, ignored).
    all_by_unit: dict[tuple[str, str], list[RequestResult]] = {}
    ok_by_unit: dict[tuple[str, str], list[RequestResult]] = {}
    for r in results:
        if not r.category:
            continue
        all_by_unit.setdefault((r.model, r.hardware), []).append(r)
        if r.ok:
            ok_by_unit.setdefault((r.model, r.hardware), []).append(r)

    rows: list[ModelRow] = []
    for (model, hardware), unit_all in all_by_unit.items():
        unit_rs = ok_by_unit.get((model, hardware), [])   # successful results only
        row = ModelRow(model, hardware)
        offloaded = _offloaded_runs_for_unit(unit_rs)
        row.offloaded_runs = len(offloaded)
        row.runs = {getattr(r, "run_label", "?") for r in unit_all}
        row.samples = len(unit_rs)
        q_vals: list[float] = []       # overall = mean of these (failed cats = 0)
        for cat in categories:
            cat_all = [r for r in unit_all if r.category == cat]
            if not cat_all:
                continue               # never attempted — no data, excluded from rank
            row.cats_attempted += 1
            cat_rs = [r for r in unit_rs if r.category == cat]   # successful only
            if not cat_rs:
                # Attempted but every request errored: a hard failure. Record it
                # as a distinct failed cell that costs the overall score (0), so a
                # model that can't do a category can't outrank one that can.
                row.cats[cat] = Cell(quality=0.0, basis="failed", failed=True,
                                     failed_n=len(cat_all))
                q_vals.append(0.0)
                continue
            row.cats_ok += 1
            # Drop this category's contribution from any run that was mostly
            # broken/truncated (a config/format bug), so it doesn't poison the
            # pooled quality; a genuine overflow (finish="length") is kept.
            by_run: dict[str, list[RequestResult]] = {}
            for r in cat_rs:
                by_run.setdefault(getattr(r, "run_label", "?"), []).append(r)
            broken_runs = {
                run for run, rr in by_run.items()
                if len(rr) >= 2 and sum(_is_broken(r) for r in rr) / len(rr) > BROKEN_RUN_FRAC
            }
            q_rs = [r for r in cat_rs if getattr(r, "run_label", "?") not in broken_runs]
            if not q_rs:                # every run broken — keep them, don't hide
                q_rs, broken_runs = cat_rs, set()

            obj = _mean([r.quality for r in q_rs if r.quality is not None])
            jud = _mean([r.judge_score for r in q_rs if r.judge_score is not None])
            eff, basis = (jud, "judged") if jud is not None else (obj, "checks")
            # throughput: drop offloaded runs
            tps = [r.tokens_per_s for r in cat_rs
                   if r.tokens_per_s and getattr(r, "run_label", "?") not in offloaded]
            row.cats[cat] = Cell(
                objective=obj, judge=jud, quality=eff, basis=basis,
                tps=_mean(tps),
                tps_std=(statistics.stdev(tps) if len(tps) >= 2 else None),
                n=len(q_rs),
                quality_excluded_runs=len(broken_runs),
            )
            if eff is not None:
                q_vals.append(eff)
        # Overall = mean over attempted-and-measurable categories, with a
        # wholly-failed category counted as 0 (so failures cost rank) and a
        # never-attempted category simply absent.
        row.overall_quality = (sum(q_vals) / len(q_vals)) if q_vals else None
        rows.append(row)

    # Rank best-first by overall quality; ties fall back to name for stability.
    return sorted(rows, key=lambda r: (-(r.overall_quality if r.overall_quality
                                         is not None else -1.0), r.model)), categories
