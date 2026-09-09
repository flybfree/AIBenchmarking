"""Model-centric scorecards and use-case fit recommendations.

Reframes the per-endpoint results around the question the operator actually
asks: *which model should I run for which use case?* Quality is a property of
the model; throughput is a property of model x hardware. A "unit" here is a
deployable (model @ endpoint/hardware) — exactly what you assign to a use case.

For each use case we combine the objective quality (reference checks) with the
LLM-judge score where available into an "effective quality", then recommend a
model per use case: the best quality, unless a faster unit reaches comparable
quality — encoding the idea that a smaller/faster model is the right pick when
its quality is good enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import Aggregate

# Two units count as "comparable quality" within this margin (on the 0-1 scale).
QUALITY_EPS = 0.05


def effective_quality(a: Aggregate) -> tuple[float | None, str]:
    """Judge score when present (subjective tasks), else the objective score."""
    if a.judge_mean is not None:
        return a.judge_mean, "judged"
    if a.quality_mean is not None:
        return a.quality_mean, "checks"
    return None, ""


@dataclass
class CatCell:
    quality: float | None
    basis: str            # "judged" | "checks" | ""
    tps: float | None
    tps_std: float | None


@dataclass
class UnitProfile:
    endpoint: str
    model: str
    hardware: str
    cats: dict[str, CatCell] = field(default_factory=dict)


@dataclass
class Fit:
    category: str
    pick_endpoint: str
    pick_model: str
    reason: str


def build_profiles(by_category: list[Aggregate]) -> tuple[list[UnitProfile], list[str]]:
    """One profile per (endpoint) unit, plus the sorted list of categories."""
    categories = sorted({a.group for a in by_category})
    units: dict[str, UnitProfile] = {}
    for a in by_category:
        u = units.setdefault(
            a.endpoint, UnitProfile(a.endpoint, a.model, a.hardware)
        )
        q, basis = effective_quality(a)
        u.cats[a.group] = CatCell(q, basis, a.tokens_per_s_mean, a.tokens_per_s_std)
    return list(units.values()), categories


def use_case_fit(by_category: list[Aggregate]) -> list[Fit]:
    by_cat: dict[str, list[Aggregate]] = {}
    for a in by_category:
        by_cat.setdefault(a.group, []).append(a)

    fits: list[Fit] = []
    for cat, aggs in sorted(by_cat.items()):
        scored = [(a, *effective_quality(a)) for a in aggs]
        scored = [(a, q, b) for (a, q, b) in scored if q is not None and a.tokens_per_s_mean]
        if not scored:
            continue
        best_q = max(scored, key=lambda t: t[1])
        fastest = max(scored, key=lambda t: t[0].tokens_per_s_mean)
        bq_a, bq_q, _ = best_q
        f_a, f_q, _ = fastest

        def pct(faster, slower):
            return 100 * (faster - slower) / slower if slower else 0.0

        if f_a.endpoint == bq_a.endpoint:
            reason = (f"best quality ({bq_q:.2f}) and fastest "
                      f"({f_a.tokens_per_s_mean:.0f} tok/s)")
            pick = bq_a
        elif f_q >= bq_q - QUALITY_EPS:
            reason = (f"comparable quality ({f_q:.2f} vs {bq_q:.2f}) at "
                      f"{pct(f_a.tokens_per_s_mean, bq_a.tokens_per_s_mean):.0f}% higher "
                      f"throughput ({f_a.tokens_per_s_mean:.0f} vs "
                      f"{bq_a.tokens_per_s_mean:.0f} tok/s)")
            pick = f_a
        else:
            reason = (f"best quality ({bq_q:.2f}); {f_a.model} is faster "
                      f"({f_a.tokens_per_s_mean:.0f} tok/s) but lower quality "
                      f"({f_q:.2f})")
            pick = bq_a
        fits.append(Fit(cat, pick.endpoint, pick.model, reason))
    return fits
