"""Head-to-head throughput comparison with a significance estimate.

For each category, compares the two fastest endpoints and reports whether the
tok/s difference is meaningful given the run-to-run variance, using the
standard error of the difference (Welch-style):

    SE = sqrt(s1^2/n1 + s2^2/n2)      z = |m1 - m2| / SE

With the small repeat counts typical here this is approximate, so we use a
conservative z threshold and always say when more repeats are needed. We also
flag any endpoint whose throughput is noisy (high coefficient of variation).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .client import RequestResult
from .metrics import Aggregate

# z = |Δ| / SE above this => call the difference significant. ~2 is nominally
# 95% for large n; we keep it here but caveat small-n in the wording.
Z_SIGNIFICANT = 2.0
# Coefficient of variation (std/mean) above this => throughput is noisy.
HIGH_CV = 0.15


@dataclass
class Comparison:
    category: str
    faster: str
    slower: str
    faster_mean: float
    slower_mean: float
    pct: float                     # how much faster, %
    z: float | None                # None when variance/counts unavailable
    significant: bool
    note: str


@dataclass
class VarianceFlag:
    endpoint: str
    category: str
    cv: float
    mean: float
    std: float


def _cv(a: Aggregate) -> float | None:
    if a.tokens_per_s_mean and a.tokens_per_s_std is not None and a.tokens_per_s_mean > 0:
        return a.tokens_per_s_std / a.tokens_per_s_mean
    return None


def variance_flags(aggs: list[Aggregate]) -> list[VarianceFlag]:
    out: list[VarianceFlag] = []
    for a in aggs:
        cv = _cv(a)
        if cv is not None and cv > HIGH_CV and a.tokens_per_s_n >= 2:
            out.append(VarianceFlag(a.endpoint, a.group, cv,
                                    a.tokens_per_s_mean, a.tokens_per_s_std))
    return out


def compare_categories(by_category: list[Aggregate]) -> list[Comparison]:
    by_cat: dict[str, list[Aggregate]] = {}
    for a in by_category:
        if a.tokens_per_s_mean is not None:
            by_cat.setdefault(a.group, []).append(a)

    out: list[Comparison] = []
    for cat, eps in sorted(by_cat.items()):
        if len(eps) < 2:
            continue
        eps.sort(key=lambda a: a.tokens_per_s_mean, reverse=True)
        top, second = eps[0], eps[1]
        m1, m2 = top.tokens_per_s_mean, second.tokens_per_s_mean
        pct = 100 * (m1 - m2) / m2 if m2 else 0.0

        z: float | None = None
        significant = False
        if (top.tokens_per_s_std is not None and second.tokens_per_s_std is not None
                and top.tokens_per_s_n >= 2 and second.tokens_per_s_n >= 2):
            se = math.sqrt(
                top.tokens_per_s_std ** 2 / top.tokens_per_s_n
                + second.tokens_per_s_std ** 2 / second.tokens_per_s_n
            )
            if se > 0:
                z = (m1 - m2) / se
                significant = z >= Z_SIGNIFICANT
                if significant:
                    note = (f"{top.endpoint} is {pct:.0f}% faster — significant "
                            f"(Δ/SE={z:.1f}).")
                else:
                    note = (f"{top.endpoint} is {pct:.0f}% faster, but that is "
                            f"within run-to-run noise (Δ/SE={z:.1f}); add repeats "
                            f"to confirm.")
            else:
                note = f"{top.endpoint} is {pct:.0f}% faster (no variance to test)."
        else:
            note = (f"{top.endpoint} is {pct:.0f}% faster, but with <2 repeats "
                    f"significance can't be estimated — raise `repeats`.")

        out.append(Comparison(
            category=cat, faster=top.endpoint, slower=second.endpoint,
            faster_mean=m1, slower_mean=m2, pct=pct, z=z,
            significant=significant, note=note,
        ))
    return out


# --- quality significance ------------------------------------------------

# Ignore quality gaps smaller than this on the 0-1 scale (not worth testing).
QUALITY_MIN_DELTA = 0.03


@dataclass
class QualityComparison:
    category: str
    leader: str
    other: str
    leader_mean: float
    other_mean: float
    z: float | None
    significant: bool
    note: str


def _eff_quality(r: RequestResult) -> float | None:
    """Judge score when the run was judged, else the objective check score."""
    if r.judge_score is not None:
        return r.judge_score
    return r.quality


def compare_quality(results: list[RequestResult]) -> list[QualityComparison]:
    """Per category, compare the top two endpoints' quality and say whether the
    gap is real or within run-to-run noise. Uses per-run effective-quality
    values (judge where available, else objective checks)."""
    by: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if not r.ok or not r.category:
            continue
        q = _eff_quality(r)
        if q is not None:
            by.setdefault(r.category, {}).setdefault(r.endpoint, []).append(q)

    out: list[QualityComparison] = []
    for cat, eps in sorted(by.items()):
        stats = {ep: vs for ep, vs in eps.items() if vs}
        if len(stats) < 2:
            continue
        means = {ep: statistics.mean(vs) for ep, vs in stats.items()}
        ordered = sorted(means, key=means.get, reverse=True)
        top, second = ordered[0], ordered[1]
        m1, m2 = means[top], means[second]
        delta = m1 - m2

        if delta < QUALITY_MIN_DELTA:
            out.append(QualityComparison(
                cat, top, second, m1, m2, 0.0, False,
                f"{top} and {second} are tied ({m1:.2f} vs {m2:.2f})."))
            continue

        n1, n2 = len(stats[top]), len(stats[second])
        z: float | None = None
        significant = False
        if n1 >= 2 and n2 >= 2:
            s1 = statistics.stdev(stats[top])
            s2 = statistics.stdev(stats[second])
            se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
            if se > 0:
                z = delta / se
                significant = z >= Z_SIGNIFICANT
                note = (f"{top} leads {m1:.2f} vs {m2:.2f} — "
                        + (f"significant (Δ/SE={z:.1f})." if significant
                           else f"within noise (Δ/SE={z:.1f}); add repeats."))
            else:
                # No variance (e.g. all identical) but means differ: real given data.
                z = float("inf")
                significant = True
                note = f"{top} leads {m1:.2f} vs {m2:.2f} — consistent (zero variance)."
        else:
            note = (f"{top} leads {m1:.2f} vs {m2:.2f}, but <2 samples — "
                    f"can't test; raise `repeats`.")
        out.append(QualityComparison(cat, top, second, m1, m2, z, significant, note))
    return out
