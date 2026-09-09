"""Cross-endpoint diagnostics.

When two or more endpoints run the *same model*, some result patterns point to
a setup problem rather than a genuine hardware/model difference. These checks
compare same-model endpoints and surface plain-language warnings:

  * offload  — one endpoint's throughput is a small fraction of a peer's on the
    same model, which usually means the model didn't fit in VRAM and spilled to
    CPU (or a much heavier quant/config is loaded).
  * ttft     — one endpoint's time-to-first-token is far higher than a peer's
    without a matching throughput gain, which usually means a server-config
    difference (request batching, context length) rather than hardware.

Thresholds are deliberately loose so an ordinary GPU-vs-GPU gap (a ~1.5-2x
bandwidth difference) does not trip them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .client import RequestResult

# A peer more than this many times faster => the slow side looks bottlenecked.
OFFLOAD_RATIO = 2.5
# TTFT this many times a peer's, without a matching throughput edge, looks like
# a config difference.
TTFT_RATIO = 1.8
# "matching throughput edge": the slow-prefill endpoint must be at least this
# many times the peer's throughput to justify its higher TTFT.
TTFT_TPS_EXCUSE = 1.3


@dataclass
class Diagnostic:
    kind: str          # "offload" | "ttft"
    endpoint: str
    model: str
    message: str


@dataclass
class _EP:
    endpoint: str
    tps: float | None
    ttft_ms: float | None


def _per_endpoint(results: list[RequestResult], model: str) -> list[_EP]:
    by_ep: dict[str, list[RequestResult]] = {}
    for r in results:
        if r.model == model and r.ok:
            by_ep.setdefault(r.endpoint, []).append(r)
    out: list[_EP] = []
    for ep, rs in by_ep.items():
        tps = [r.tokens_per_s for r in rs if r.tokens_per_s]
        ttft = [r.ttft_s * 1000 for r in rs if r.ttft_s is not None]
        out.append(_EP(
            endpoint=ep,
            tps=statistics.mean(tps) if tps else None,
            ttft_ms=statistics.mean(ttft) if ttft else None,
        ))
    return out


def diagnose(results: list[RequestResult]) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    models = sorted({r.model for r in results if r.ok})
    for model in models:
        eps = _per_endpoint(results, model)
        if len(eps) < 2:
            continue

        # --- offload / underperformance ---
        tps_eps = [e for e in eps if e.tps]
        if len(tps_eps) >= 2:
            best = max(tps_eps, key=lambda e: e.tps)
            for e in tps_eps:
                if e is best:
                    continue
                if best.tps >= OFFLOAD_RATIO * e.tps:
                    pct = 100 * e.tps / best.tps
                    out.append(Diagnostic(
                        kind="offload",
                        endpoint=e.endpoint,
                        model=model,
                        message=(
                            f"{e.endpoint} runs {model} at {e.tps:.1f} tok/s - only "
                            f"{pct:.0f}% of {best.endpoint}'s {best.tps:.1f} tok/s on "
                            f"the same model. Likely the model doesn't fit in VRAM and "
                            f"is offloading to CPU (or a heavier quant/config is loaded). "
                            f"Check GPU offload / model size on {e.endpoint}."
                        ),
                    ))

        # --- TTFT out of line with throughput ---
        ttft_eps = [e for e in eps if e.ttft_ms and e.tps]
        if len(ttft_eps) >= 2:
            fastest_prefill = min(ttft_eps, key=lambda e: e.ttft_ms)
            for e in ttft_eps:
                if e is fastest_prefill:
                    continue
                higher_ttft = e.ttft_ms >= TTFT_RATIO * fastest_prefill.ttft_ms
                justified = e.tps >= TTFT_TPS_EXCUSE * fastest_prefill.tps
                if higher_ttft and not justified:
                    ratio = e.ttft_ms / fastest_prefill.ttft_ms
                    out.append(Diagnostic(
                        kind="ttft",
                        endpoint=e.endpoint,
                        model=model,
                        message=(
                            f"{e.endpoint} time-to-first-token ({e.ttft_ms:.0f} ms) is "
                            f"{ratio:.1f}x {fastest_prefill.endpoint}'s "
                            f"({fastest_prefill.ttft_ms:.0f} ms) on the same model, but "
                            f"throughput is comparable ({e.tps:.1f} vs "
                            f"{fastest_prefill.tps:.1f} tok/s). Points to a server-config "
                            f"difference (request batching, context length) rather than "
                            f"hardware; match {e.endpoint}'s settings to "
                            f"{fastest_prefill.endpoint} for a fair comparison."
                        ),
                    ))
    return out
