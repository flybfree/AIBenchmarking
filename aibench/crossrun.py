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
import re
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


@dataclass
class Rec:
    category: str
    model: str
    quality: float
    basis: str            # "judged" | "checks"
    tps: float | None
    runs: int
    note: str


def per_machine_recommendations(
    rows: list[ModelRow], categories: list[str]
) -> dict[str, list[Rec]]:
    """For each hardware/endpoint, the model to run for each use case.

    The goal is per-machine deployment: which model best serves each use case
    ON THAT MACHINE. Picks the best-quality model, unless a faster one on the
    same machine reaches comparable quality (within QUALITY_EPS) — the smaller/
    faster model is the right pick when it's good enough. Never recommends a
    model that FAILED the category. Returns {hardware: [Rec, ...]}.
    """
    from .scorecard import QUALITY_EPS

    by_hw: dict[str, list[ModelRow]] = {}
    for r in rows:
        by_hw.setdefault(r.hardware, []).append(r)

    out: dict[str, list[Rec]] = {}
    for hw, units in by_hw.items():
        recs: list[Rec] = []
        for cat in categories:
            cands = [
                (u, u.cats[cat]) for u in units
                if cat in u.cats and u.cats[cat].quality is not None
                and not u.cats[cat].failed
            ]
            if not cands:
                continue
            best_u, best_c = max(cands, key=lambda t: t[1].quality)
            # "Good enough" = within QUALITY_EPS of the best quality on this
            # machine. Among those, take the FASTEST — not the global-fastest
            # candidate, which may be well below the quality bar. (Picking the
            # single fastest and only then checking its quality wrongly fell
            # back to a slow top-quality model whenever the very fastest model
            # happened to be low quality, ignoring fast models tied at the top.)
            good_enough = [(u, c) for (u, c) in cands
                           if c.quality >= best_c.quality - QUALITY_EPS]
            with_tps = [(u, c) for (u, c) in good_enough if c.tps]
            if with_tps:
                pick_u, pick_c = max(with_tps, key=lambda t: t[1].tps)
            else:
                pick_u, pick_c = best_u, best_c
            if pick_u is best_u:
                note = (f"best quality ({best_c.quality:.2f})"
                        + (f" @ {pick_c.tps:.0f} tok/s" if pick_c.tps else ""))
            else:
                vs = f" vs {best_c.tps:.0f}" if best_c.tps else ""
                note = (f"comparable quality ({pick_c.quality:.2f} vs {best_c.quality:.2f}) "
                        f"at higher speed ({pick_c.tps:.0f}{vs} tok/s)")
            recs.append(Rec(cat, pick_u.model, pick_c.quality, pick_c.basis,
                            pick_c.tps, len(pick_u.runs), note))
        out[hw] = recs
    return out


# --- single-default-per-machine selection ------------------------------------

@dataclass
class MachineDefault:
    hardware: str
    model: str
    overall: float           # this pick's overall quality
    top_overall: float       # best overall quality available on the machine
    tps: float | None        # representative throughput (mean per-category tok/s)
    runs: int
    note: str


@dataclass
class ComboPick:
    hardware: str
    model: str
    overall: float
    tps: float | None
    runs: int
    covers: list[str]        # use cases this box's default is the fleet's best at


@dataclass
class Combination:
    picks: list[ComboPick]
    coverage: float          # mean over use cases of the best quality the pair reaches
    per_use: dict            # cat -> (hardware, model, quality) that covers it


def _unit_tps(row: ModelRow) -> float | None:
    """A single representative throughput for a unit: mean of its per-category
    tok/s. Rough (short-output categories are TTFT-bound) but comparable across
    units, enough to break quality ties toward the faster model."""
    vals = [c.tps for c in row.cats.values() if c.tps]
    return sum(vals) / len(vals) if vals else None


def best_default_per_machine(rows: list[ModelRow]) -> dict[str, MachineDefault]:
    """The single best all-round DEFAULT model to leave loaded on each machine.

    Balanced: rank by overall quality, but prefer a faster model when it's within
    QUALITY_EPS of the machine's best overall — the fastest all-rounder that's
    still excellent. Returns {hardware: MachineDefault}."""
    from .scorecard import QUALITY_EPS
    by_hw: dict[str, list[ModelRow]] = {}
    for r in rows:
        if r.overall_quality is not None:
            by_hw.setdefault(r.hardware, []).append(r)

    out: dict[str, MachineDefault] = {}
    for hw, us in by_hw.items():
        best = max(us, key=lambda u: u.overall_quality)
        good = [u for u in us if u.overall_quality >= best.overall_quality - QUALITY_EPS]
        good_tps = [u for u in good if _unit_tps(u)]
        pick = max(good_tps, key=lambda u: _unit_tps(u)) if good_tps else best
        tps = _unit_tps(pick)
        if pick is best:
            note = (f"top all-round quality ({pick.overall_quality:.3f})"
                    + (f" · {tps:.0f} tok/s" if tps else ""))
        else:
            note = (f"within {QUALITY_EPS:.2f} of the best "
                    f"({pick.overall_quality:.3f} vs {best.overall_quality:.3f}) "
                    f"and faster ({tps:.0f} tok/s)")
        out[hw] = MachineDefault(hw, pick.model, pick.overall_quality,
                                 best.overall_quality, tps, len(pick.runs), note)
    return out


def best_complementary_combo(
    rows: list[ModelRow], categories: list[str]
) -> Combination | None:
    """Pick one default per machine so the pair, TOGETHER, covers every use case
    best — route each use case to whichever box's default is stronger at it.

    Fleet coverage = mean over use cases of max(quality across the chosen units).
    This rewards complementary strengths (e.g. a fast reasoning/code model on one
    box, a quality writer on the other). Balanced: among combinations within a
    small margin of the best coverage, prefer the faster one (higher total tok/s).
    """
    import itertools
    from .scorecard import QUALITY_EPS

    by_hw: dict[str, list[ModelRow]] = {}
    for r in rows:
        if r.overall_quality is not None:
            by_hw.setdefault(r.hardware, []).append(r)
    hws = sorted(by_hw)
    if not hws:
        return None
    # Cap the search: keep each machine's top candidates by overall quality.
    cand = {hw: sorted(us, key=lambda u: u.overall_quality, reverse=True)[:12]
            for hw, us in by_hw.items()}

    def cell_q(u: ModelRow, c: str) -> float:
        cc = u.cats.get(c)
        if not cc or cc.failed or cc.quality is None:
            return 0.0
        return cc.quality

    def route(combo, c):
        """Which unit serves use case c: the best-quality one, but prefer the
        faster box when it's within QUALITY_EPS — so bulk work lands on the fast
        machine and only what it's genuinely weaker at goes to the other."""
        qs = [(u, cell_q(u, c)) for u in combo]
        best_q = max(q for _, q in qs)
        good = [(u, q) for (u, q) in qs if q >= best_q - QUALITY_EPS]
        return max(good, key=lambda t: _unit_tps(t[0]) or 0)  # (unit, quality)

    # Score every combination on three tiers, applied in order so a strength in
    # one use case can't average away a weakness in another. Coverage uses the
    # balanced routing above, so a fast box that's "good enough" is used rather
    # than funneling everything to one slow high-quality model:
    #   1. worst-covered use case (maximin) — "cover EVERY use case"
    #   2. mean coverage — overall strength
    #   3. total throughput — balanced tiebreak toward the faster pair
    scored = []
    for combo in itertools.product(*(cand[hw] for hw in hws)):
        per = [route(combo, c)[1] for c in categories]
        scored.append((min(per), sum(per) / len(per),
                       sum(_unit_tps(u) or 0 for u in combo), combo))
    if not scored:
        return None
    top_min = max(s[0] for s in scored)
    f1 = [s for s in scored if s[0] >= top_min - QUALITY_EPS]
    top_mean = max(s[1] for s in f1)
    f2 = [s for s in f1 if s[1] >= top_mean - QUALITY_EPS]
    best_combo = max(f2, key=lambda s: s[2])[3]

    # Attribute each use case to the box that serves it under balanced routing.
    per_use: dict = {}
    covers: dict[str, list[str]] = {u.hardware: [] for u in best_combo}
    for c in categories:
        unit, q = route(best_combo, c)
        per_use[c] = (unit.hardware, unit.model, q)
        covers[unit.hardware].append(c)
    picks = [ComboPick(u.hardware, u.model, u.overall_quality, _unit_tps(u),
                       len(u.runs), covers[u.hardware]) for u in best_combo]
    coverage = sum(q for _, _, q in per_use.values()) / len(categories)
    return Combination(picks, coverage, per_use)


# --- size / efficiency standouts ---------------------------------------------

# Dense parameter count (billions) that fits comfortably in modest VRAM — the
# ceiling for a "small-footprint" model.
SMALL_FOOTPRINT_B = 14.0


def parse_param_size(model: str) -> tuple[float | None, float | None]:
    """Best-effort (total_B, active_B) parsed from a model id — the benchmark
    stores no parameter count, so we read it from the name. total is the largest
    "<N>b" token (footprint); active is the smallest "a<N>b" token (MoE active
    params). Either may be None when the name doesn't say."""
    name = model.split("/")[-1].lower()
    allb = re.findall(r"(\d+(?:\.\d+)?)b", name)
    total = max((float(x) for x in allb), default=None)
    act = re.findall(r"a(\d+(?:\.\d+)?)b", name)
    active = min((float(x) for x in act), default=None)
    return total, active


@dataclass
class SizeStandout:
    model: str
    hardware: str
    total_b: float
    active_b: float | None
    overall: float
    runs: int


def small_model_standouts(
    rows: list[ModelRow], max_total_b: float = SMALL_FOOTPRINT_B
) -> tuple[list[SizeStandout], list[SizeStandout]]:
    """Two efficiency lenses the raw ranking hides:

    - small-footprint standouts: dense models <= max_total_b params, ranked by
      overall quality (best quality you can run in little VRAM);
    - compute-efficient MoEs: models whose *active* params are a small fraction
      of total (fast decode despite a large footprint).

    Returns (small_footprint, efficient_moe), each sorted best-quality first.
    Models whose size can't be parsed from the name are skipped."""
    small, moe = [], []
    for r in rows:
        if r.overall_quality is None:
            continue
        total, active = parse_param_size(r.model)
        if total is None:
            continue
        s = SizeStandout(r.model, r.hardware, total, active,
                         r.overall_quality, len(r.runs))
        if total <= max_total_b:
            small.append(s)
        elif active is not None and active <= total / 3:
            moe.append(s)
    small.sort(key=lambda s: s.overall, reverse=True)
    moe.sort(key=lambda s: s.overall, reverse=True)
    return small, moe
