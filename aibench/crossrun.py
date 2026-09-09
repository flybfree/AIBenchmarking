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


def aggregate_by_model(results: list[RequestResult]) -> tuple[list[ModelRow], list[str]]:
    """Pool results into (model @ hardware) rows with per-category cells."""
    categories = sorted({r.category for r in results if r.category})
    rows: dict[tuple[str, str], ModelRow] = {}
    # group results by (model, hardware, category)
    buckets: dict[tuple[str, str, str], list[RequestResult]] = {}
    for r in results:
        if not r.ok or not r.category:
            continue
        buckets.setdefault((r.model, r.hardware, r.category), []).append(r)

    for (model, hardware, cat), rs in buckets.items():
        row = rows.setdefault((model, hardware), ModelRow(model, hardware))
        tps = [r.tokens_per_s for r in rs if r.tokens_per_s]
        obj = _mean([r.quality for r in rs if r.quality is not None])
        jud = _mean([r.judge_score for r in rs if r.judge_score is not None])
        eff, basis = (jud, "judged") if jud is not None else (obj, "checks")
        row.cats[cat] = Cell(
            objective=obj, judge=jud, quality=eff, basis=basis,
            tps=_mean(tps),
            tps_std=(statistics.stdev(tps) if len(tps) >= 2 else None),
            n=len(rs),
        )
        row.runs.update(getattr(r, "run_label", "?") for r in rs)
        row.samples += len(rs)

    ordered = sorted(rows.values(), key=lambda r: (r.model, r.hardware))
    return ordered, categories
