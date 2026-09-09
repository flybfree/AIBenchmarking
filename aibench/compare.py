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
from dataclasses import dataclass

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
