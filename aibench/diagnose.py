"""Cross-endpoint diagnostics.

When two or more endpoints run the *same model*, some result patterns point to
a setup problem rather than a genuine hardware/model difference:

  * offload  — one endpoint's throughput is a small fraction of a peer's on the
    same model, usually meaning the model didn't fit in VRAM and spilled to CPU
    (or a heavier quant/config is loaded).
  * ttft     — one endpoint's time-to-first-token is far higher than a peer's
    without a matching throughput gain, usually a server-config difference
    (request batching, context length) or extra load, not hardware.

Comparisons are done **per use case (category)** and use the **median** across
repeats, so an anomaly concentrated in a few categories isn't hidden by pooling,
and a single slow outlier doesn't skew a mean. Findings are then coalesced into
one message per (endpoint, kind) that lists every affected use case. Thresholds
stay loose so an ordinary GPU-vs-GPU gap (~1.5-2x) doesn't trip them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from statistics import median

from .client import RequestResult

OFFLOAD_RATIO = 2.5     # peer this many x faster => slow side looks bottlenecked
TTFT_RATIO = 1.8        # TTFT this many x a peer's => suspicious
TTFT_TPS_EXCUSE = 1.3   # ...unless throughput is at least this many x the peer's
# Single-endpoint CPU-offload heuristic: generation that is both LOW and FLAT
# (near-constant tok/s regardless of task) is the signature of CPU-bound decode
# because the model doesn't fully fit in VRAM. Modern consumer GPUs run these
# models well above this floor when fully resident, so a flat median below it is
# suspicious on its own — no same-model peer required.
OFFLOAD_TPS_FLOOR = 25.0
OFFLOAD_FLAT_CV = 0.15


@dataclass
class Diagnostic:
    kind: str          # "offload" | "ttft"
    endpoint: str
    model: str
    message: str
    categories: list[str] = field(default_factory=list)


def _by_model_category(
    results: list[RequestResult], model: str
) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """{category: {endpoint: (median tok/s, median TTFT ms)}} for one model."""
    raw: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in results:
        if r.model != model or not r.ok or not r.category:
            continue
        slot = raw.setdefault(r.category, {}).setdefault(
            r.endpoint, {"tps": [], "ttft": []}
        )
        if r.tokens_per_s:
            slot["tps"].append(r.tokens_per_s)
        if r.ttft_s is not None:
            slot["ttft"].append(r.ttft_s * 1000)
    out: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    for cat, eps in raw.items():
        out[cat] = {
            ep: (median(v["tps"]) if v["tps"] else None,
                 median(v["ttft"]) if v["ttft"] else None)
            for ep, v in eps.items()
        }
    return out


def diagnose(results: list[RequestResult]) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    models = sorted({r.model for r in results if r.ok})
    for model in models:
        cats = _by_model_category(results, model)
        # endpoint -> list of per-category hits
        offload: dict[str, list[tuple]] = {}
        ttft: dict[str, list[tuple]] = {}

        for cat, eps in cats.items():
            tps = {ep: t for ep, (t, _) in eps.items() if t}
            if len(tps) >= 2:
                best_ep = max(tps, key=tps.get)
                for ep, t in tps.items():
                    if ep != best_ep and tps[best_ep] >= OFFLOAD_RATIO * t:
                        offload.setdefault(ep, []).append((cat, t, tps[best_ep], best_ep))

            pref = {ep: (tt, t) for ep, (t, tt) in eps.items() if tt and t}
            if len(pref) >= 2:
                fast_ep = min(pref, key=lambda e: pref[e][0])
                f_ttft, f_tps = pref[fast_ep]
                for ep, (tt, t) in pref.items():
                    if ep != fast_ep and tt >= TTFT_RATIO * f_ttft and t < TTFT_TPS_EXCUSE * f_tps:
                        ttft.setdefault(ep, []).append((cat, tt, f_ttft, fast_ep, t, f_tps))

        for ep, hits in offload.items():
            names = sorted(h[0] for h in hits)
            cat, t, best_tps, best_ep = min(hits, key=lambda h: h[1] / h[2])
            pct = 100 * t / best_tps
            out.append(Diagnostic(
                "offload", ep, model,
                f"{ep} underperforms {best_ep} on {model} in {len(names)} use case(s) "
                f"({', '.join(names)}). Worst: {cat} at {t:.0f} tok/s = {pct:.0f}% of "
                f"{best_ep}'s {best_tps:.0f}. Likely VRAM offload to CPU or a heavier "
                f"quant/config; check GPU offload and model load on {ep}.",
                categories=names,
            ))

        for ep, hits in ttft.items():
            names = sorted(h[0] for h in hits)
            cat, tt, f_ttft, fast_ep, t, f_tps = max(hits, key=lambda h: h[1] / h[2])
            ratio = tt / f_ttft
            out.append(Diagnostic(
                "ttft", ep, model,
                f"{ep} time-to-first-token runs up to {ratio:.1f}x {fast_ep}'s on "
                f"{model} in {len(names)} use case(s) ({', '.join(names)}) without a "
                f"matching throughput gain. Worst: {cat}, {tt:.0f} vs {f_ttft:.0f} ms "
                f"at {t:.0f} vs {f_tps:.0f} tok/s. Points to a server-config difference "
                f"(batching, context length) or load on {ep}, not hardware; match its "
                f"settings to {fast_ep}.",
                categories=names,
            ))

    # Single-endpoint offload: a flat, low throughput profile on its own (works
    # even when endpoints run different models, where the same-model check above
    # doesn't apply). Skip endpoints already flagged offload above.
    already = {d.endpoint for d in out if d.kind == "offload"}
    out.extend(d for d in offload_flags(results) if d.endpoint not in already)
    return out


def offload_flags(results: list[RequestResult]) -> list[Diagnostic]:
    by_ep: dict[str, list[float]] = {}
    model_of: dict[str, str] = {}
    for r in results:
        if r.ok and r.tokens_per_s:
            by_ep.setdefault(r.endpoint, []).append(r.tokens_per_s)
            model_of.setdefault(r.endpoint, r.model)
    out: list[Diagnostic] = []
    for ep, tps in by_ep.items():
        if len(tps) < 3:
            continue
        mean = statistics.mean(tps)
        med = statistics.median(tps)
        cv = statistics.stdev(tps) / mean if mean else 1.0
        if med < OFFLOAD_TPS_FLOOR and cv < OFFLOAD_FLAT_CV:
            model = model_of.get(ep, "")
            out.append(Diagnostic(
                "offload", ep, model,
                f"{ep} shows flat, low throughput (median {med:.0f} tok/s, "
                f"CV {cv*100:.0f}%) running {model} — the signature of CPU offload "
                f"(the model likely doesn't fully fit in VRAM). Its throughput "
                f"numbers aren't representative; free VRAM or use a smaller quant "
                f"on {ep} and re-run. (Quality is unaffected.)",
            ))
    return out
