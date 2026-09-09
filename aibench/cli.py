"""Command-line interface for aibench.

Commands:
  run          Run a benchmark from a config file, save results, build a report.
  list-tasks   List built-in task suites and their tasks.
  report       Regenerate an HTML report from a saved results JSON.
  ping         Quick connectivity/latency check against configured endpoints.
  init-config  Write a starter config file you can edit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import RunConfig
from .metrics import aggregate, aggregate_by_category
from .report import render
from .runner import run
from .storage import save_run, load_run
from .tasks import ALL_SUITES

try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()

    def out(msg: str = "") -> None:
        _console.print(msg)
except Exception:  # rich optional
    _console = None

    def out(msg: str = "") -> None:
        print(msg)


_EXAMPLE_CONFIG = """\
# aibench run configuration
label: "bakeoff"
repeats: 3          # measured runs per (endpoint, task)
warmup: 1           # discarded warmup runs
timeout_s: 300
concurrency: 1      # >1 measures throughput under concurrent load
# max_tokens: 512      # answer-token budget (overrides per-task defaults)
reasoning_reserve: 8192  # extra tokens for thinking, added on top of the answer
                         # budget so reasoning models don't run out mid-thought
                         # (heavy reasoners can spend 4k+ tokens before answering)

tasks:
  - all             # or: creative_writing, code_generation, summarization, tool_use

endpoints:
  - name: "rig1-llama3-8b"
    base_url: "http://192.168.1.50:11434/v1"   # Ollama example
    model: "llama3:8b"
    hardware: "RTX 4090"

  - name: "rig2-qwen2.5-7b"
    base_url: "http://192.168.1.51:1234/v1"     # LM Studio example
    model: "qwen2.5-7b-instruct"
    hardware: "M2 Max"
"""


def _cmd_list_tasks(_args) -> int:
    for suite in ALL_SUITES.values():
        out(f"\n[bold]{suite.name}[/bold] — {suite.description}"
            if _console else f"\n{suite.name} — {suite.description}")
        for t in suite.tasks:
            mt = t.params.get("max_tokens", "-")
            out(f"    {t.id:22} ({t.category})  max_tokens={mt}")
    return 0


def _cmd_init_config(args) -> int:
    p = Path(args.path)
    if p.exists() and not args.force:
        out(f"Refusing to overwrite existing {p} (use --force).")
        return 1
    p.write_text(_EXAMPLE_CONFIG, encoding="utf-8")
    out(f"Wrote starter config to {p}. Edit the endpoints, then:\n"
        f"    python -m aibench run --config {p}")
    return 0


def _build_report(payload: dict, out_html: Path) -> Path:
    from .client import RequestResult
    from .diagnose import diagnose
    from .compare import compare_categories, variance_flags
    results = [RequestResult(**r) for r in payload["results"]]
    by_task = aggregate(results, group_key="task")
    by_cat = aggregate_by_category(results)
    cfg = payload.get("config", {})
    meta = {
        "label": cfg.get("label", ""),
        "started_at": payload.get("started_at", ""),
        "finished_at": payload.get("finished_at", ""),
        "tasks": ", ".join(cfg.get("tasks", [])),
        "repeats": cfg.get("repeats", ""),
        "concurrency": cfg.get("concurrency", ""),
    }
    return render(
        by_task, by_cat, meta, out_html,
        results=results, diagnostics=diagnose(results),
        endpoints=cfg.get("endpoints", []),
        comparisons=compare_categories(by_cat),
        variance=variance_flags(by_cat),
    )


def _print_diagnostics(results) -> None:
    from .diagnose import diagnose
    diags = diagnose(results)
    if not diags:
        return
    out("\nDiagnostics (same-model endpoint comparison):")
    for d in diags:
        out(f"  [!] {d.endpoint} ({d.kind}): {d.message}")


def _print_comparison(by_cat) -> None:
    from .compare import compare_categories
    comps = compare_categories(by_cat)
    if not comps:
        return
    out("\nHead-to-head throughput (fastest per category):")
    for c in comps:
        verdict = ("significant" if c.significant
                   else ("within noise" if c.z is not None else "need >=2 repeats"))
        z = f"Δ/SE={c.z:.1f}" if c.z is not None else ""
        out(f"  {c.category:16} {c.faster} {c.faster_mean:.1f} vs "
            f"{c.slower} {c.slower_mean:.1f} tok/s  (+{c.pct:.0f}%, {verdict} {z})")


def _print_summary(by_cat) -> None:
    if _console:
        table = Table(title="Per-category summary")
        for col in ("Endpoint", "Hardware", "Category", "tok/s", "TTFT ms", "quality", "ok/n"):
            table.add_column(col, justify="right" if col in ("tok/s", "TTFT ms", "quality") else "left")
        for a in by_cat:
            table.add_row(
                a.endpoint, a.hardware, a.group,
                f"{a.tokens_per_s_mean:.1f}" if a.tokens_per_s_mean else "—",
                f"{a.ttft_ms_mean:.0f}" if a.ttft_ms_mean else "—",
                f"{a.quality_mean:.2f}" if a.quality_mean is not None else "—",
                f"{a.n_ok}/{a.n}",
            )
        _console.print(table)
    else:
        for a in by_cat:
            tps = f"{a.tokens_per_s_mean:.1f}" if a.tokens_per_s_mean else "—"
            out(f"{a.endpoint:20} {a.group:18} {tps:>8} tok/s  ok {a.n_ok}/{a.n}")


def _cmd_run(args) -> int:
    cfg = RunConfig.load(args.config)
    if args.out:
        cfg.output_dir = args.out
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.reasoning_reserve is not None:
        cfg.reasoning_reserve = args.reasoning_reserve
    if args.repeats is not None:
        cfg.repeats = args.repeats
    if args.parallel_endpoints is not None:
        cfg.parallel_endpoints = args.parallel_endpoints
    out(f"Running benchmark: {len(cfg.endpoints)} endpoint(s), "
        f"tasks={cfg.tasks}, repeats={cfg.repeats}\n")

    output = asyncio.run(run(cfg, progress=out))

    if not args.no_score:
        from .scoring.run import score_results
        n = score_results(output.results)
        out(f"\nScored {n} results (reference-based quality checks).")

    json_path = save_run(output, cfg.output_dir)
    out(f"Saved raw results -> {json_path}")

    payload = load_run(json_path)
    from .client import RequestResult
    results = [RequestResult(**r) for r in payload["results"]]
    by_cat = aggregate_by_category(results)
    out("")
    _print_summary(by_cat)
    _print_diagnostics(results)
    _print_comparison(by_cat)

    html_path = json_path.with_suffix(".html")
    _build_report(payload, html_path)
    out(f"\nReport -> {html_path}")
    if args.open:
        import webbrowser
        webbrowser.open(html_path.resolve().as_uri())
    return 0


def _cmd_report(args) -> int:
    payload = load_run(args.results)
    out_html = Path(args.out) if args.out else Path(args.results).with_suffix(".html")
    _build_report(payload, out_html)
    out(f"Report -> {out_html}")
    if args.open:
        import webbrowser
        webbrowser.open(out_html.resolve().as_uri())
    return 0


def _cmd_score(args) -> int:
    """(Re)score a saved results JSON in place and rebuild its report."""
    from .scoring.run import score_results
    from .client import RequestResult

    payload = load_run(args.results)
    results = [RequestResult(**r) for r in payload["results"]]
    # Recompute derived timing (e.g. burst detection) with the current logic,
    # then apply quality scoring.
    for r in results:
        r.finalize()
    n = score_results(results)
    out(f"Scored {n} of {len(results)} results.")

    # Persist scores back into the same JSON so they travel with the run.
    payload["results"] = [asdict(r) for r in results]
    Path(args.results).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    by_cat = aggregate_by_category(results)
    out("")
    _print_summary(by_cat)
    _print_diagnostics(results)
    _print_comparison(by_cat)

    out_html = Path(args.out) if args.out else Path(args.results).with_suffix(".html")
    _build_report(payload, out_html)
    out(f"\nReport -> {out_html}")
    if args.open:
        import webbrowser
        webbrowser.open(out_html.resolve().as_uri())
    return 0


def _cmd_ping(args) -> int:
    import time
    import httpx

    cfg = RunConfig.load(args.config)

    async def _ping_all() -> int:
        rc = 0
        async with httpx.AsyncClient() as client:
            for ep in cfg.endpoints:
                url = ep.chat_url
                body = {
                    "model": ep.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "stream": False,
                }
                headers = {"Content-Type": "application/json"}
                if ep.api_key and ep.api_key != "not-needed":
                    headers["Authorization"] = f"Bearer {ep.api_key}"
                t0 = time.perf_counter()
                try:
                    r = await client.post(url, json=body, headers=headers, timeout=15)
                    dt = (time.perf_counter() - t0) * 1000
                    status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
                    out(f"  {ep.name:24} {status:10} {dt:6.0f} ms  {url}")
                    if r.status_code != 200:
                        rc = 1
                except Exception as e:
                    out(f"  {ep.name:24} FAIL       {type(e).__name__}: {e}")
                    rc = 1
        return rc

    return asyncio.run(_ping_all())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aibench", description="Local AI inference benchmarking.")
    p.add_argument("--version", action="version", version=f"aibench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run a benchmark from a config file.")
    r.add_argument("--config", "-c", required=True)
    r.add_argument("--out", "-o", help="Output directory for results/report.")
    r.add_argument("--max-tokens", type=int, dest="max_tokens",
                   help="Answer-token budget (overrides per-task defaults). "
                        "Reasoning room is added on top via --reasoning-reserve.")
    r.add_argument("--reasoning-reserve", type=int, dest="reasoning_reserve",
                   help="Extra output tokens added on top of the answer budget "
                        "so reasoning models have room to think (default 8192).")
    r.add_argument("--repeats", type=int, help="Override measured runs per task.")
    r.add_argument("--parallel-endpoints", dest="parallel_endpoints",
                   action="store_true", default=None,
                   help="Benchmark all endpoints at once. Only for endpoints on "
                        "SEPARATE machines (shared-GPU endpoints would skew timings).")
    r.add_argument("--no-score", action="store_true",
                   help="Skip Phase 2 reference-based quality scoring.")
    r.add_argument("--open", action="store_true", help="Open the HTML report when done.")
    r.set_defaults(func=_cmd_run)

    lt = sub.add_parser("list-tasks", help="List built-in task suites.")
    lt.set_defaults(func=_cmd_list_tasks)

    rp = sub.add_parser("report", help="Rebuild an HTML report from results JSON.")
    rp.add_argument("results")
    rp.add_argument("--out", "-o")
    rp.add_argument("--open", action="store_true")
    rp.set_defaults(func=_cmd_report)

    sc = sub.add_parser("score", help="(Re)score a saved results JSON and rebuild its report.")
    sc.add_argument("results")
    sc.add_argument("--out", "-o")
    sc.add_argument("--open", action="store_true")
    sc.set_defaults(func=_cmd_score)

    pg = sub.add_parser("ping", help="Check connectivity to configured endpoints.")
    pg.add_argument("--config", "-c", required=True)
    pg.set_defaults(func=_cmd_ping)

    ic = sub.add_parser("init-config", help="Write a starter config file.")
    ic.add_argument("path", nargs="?", default="configs/example.yaml")
    ic.add_argument("--force", action="store_true")
    ic.set_defaults(func=_cmd_init_config)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
