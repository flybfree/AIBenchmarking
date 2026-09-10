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
    basis: str = ""                    # "judged" | "checks"
    tps: float | None = None
    tps_std: float | None = None
    n: int = 0                         # measured samples pooled


@dataclass
class ModelRow:
    model: str
    hardware: str
    cats: dict[str, Cell] = field(default_factory=dict)
    runs: set[str] = field(default_factory=set)
    samples: int = 0
    offloaded_runs: int = 0            # runs excluded from throughput pooling


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
    by_unit: dict[tuple[str, str], list[RequestResult]] = {}
    for r in results:
        if r.ok and r.category:
            by_unit.setdefault((r.model, r.hardware), []).append(r)

    rows: list[ModelRow] = []
    for (model, hardware), unit_rs in by_unit.items():
        row = ModelRow(model, hardware)
        offloaded = _offloaded_runs_for_unit(unit_rs)
        row.offloaded_runs = len(offloaded)
        row.runs = {getattr(r, "run_label", "?") for r in unit_rs}
        row.samples = len(unit_rs)
        for cat in categories:
            cat_rs = [r for r in unit_rs if r.category == cat]
            if not cat_rs:
                continue
            obj = _mean([r.quality for r in cat_rs if r.quality is not None])
            jud = _mean([r.judge_score for r in cat_rs if r.judge_score is not None])
            eff, basis = (jud, "judged") if jud is not None else (obj, "checks")
            # throughput: drop offloaded runs
            tps = [r.tokens_per_s for r in cat_rs
                   if r.tokens_per_s and getattr(r, "run_label", "?") not in offloaded]
            row.cats[cat] = Cell(
                objective=obj, judge=jud, quality=eff, basis=basis,
                tps=_mean(tps),
                tps_std=(statistics.stdev(tps) if len(tps) >= 2 else None),
                n=len(cat_rs),
            )
        rows.append(row)

    return sorted(rows, key=lambda r: (r.model, r.hardware)), categories
