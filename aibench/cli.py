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
  - all             # creative_writing, code_generation, summarization, tool_use, agents

# Optional Phase 3 LLM-judge: scores subjective tasks (creative, summaries) on a
# 1-5 rubric so quality actually separates models. Point it at a strong model;
# it need not be one of the endpoints under test. Enable at runtime with --judge.
#
# Local judge (no key):
# judge:
#   name: "judge"
#   base_url: "http://192.168.3.89:1234/v1"
#   model: "some-strong-model"
#
# External provider (any OpenAI-compatible API). Put the key in an ENV VAR, not
# here — `env:NAME` is read from the environment at runtime. NOTE: this sends the
# models' outputs (and task prompts, incl. the source articles) to that provider.
# judge:
#   name: "judge"
#   base_url: "https://api.openai.com/v1"      # or https://openrouter.ai/api/v1, etc.
#   model: "gpt-4o-mini"
#   api_key: "env:OPENAI_API_KEY"              # set OPENAI_API_KEY in your shell

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
    from .compare import compare_categories, variance_flags, compare_quality
    from .scorecard import build_profiles, use_case_fit
    results = [RequestResult(**r) for r in payload["results"]]
    by_task = aggregate(results, group_key="task")
    by_cat = aggregate_by_category(results)
    profiles, categories = build_profiles(by_cat)
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
        profiles=profiles, categories=categories,
        fits=use_case_fit(by_cat),
        quality_comparisons=compare_quality(results),
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
        if getattr(c, "unreliable", False):
            out(f"  {c.category:16} {c.faster} {c.faster_mean:.1f} vs "
                f"{c.slower} {c.slower_mean:.1f} tok/s  "
                f"(n/a — outputs too short for reliable tok/s; compare TTFT)")
            continue
        verdict = ("significant" if c.significant
                   else ("within noise" if c.z is not None else "need >=2 repeats"))
        z = f"Δ/SE={c.z:.1f}" if c.z is not None else ""
        out(f"  {c.category:16} {c.faster} {c.faster_mean:.1f} vs "
            f"{c.slower} {c.slower_mean:.1f} tok/s  (+{c.pct:.0f}%, {verdict} {z})")


def _print_quality_comparison(results) -> None:
    from .compare import compare_quality
    comps = compare_quality(results)
    real = [c for c in comps if c.significant]
    if not real:
        out("\nQuality head-to-head: no significant quality differences "
            "(same-model gaps are within noise).")
        return
    out("\nHead-to-head quality (significant differences only):")
    for c in real:
        out(f"  {c.category:16} {c.leader} {c.leader_mean:.2f} vs "
            f"{c.other} {c.other_mean:.2f}  ({c.note})")


def _print_summary(by_cat) -> None:
    if _console:
        table = Table(title="Per-category summary")
        num_cols = ("tok/s", "agg", "TTFT ms", "p95", "spikes", "runaway", "quality")
        for col in ("Endpoint", "Hardware", "Category", "tok/s", "agg", "TTFT ms", "p95", "spikes", "runaway", "quality", "ok/n"):
            table.add_column(col, justify="right" if col in num_cols else "left")
        for a in by_cat:
            if a.tokens_per_s_mean is None:
                tps = "—"
            elif getattr(a, "tps_unreliable", False):
                tps = f"~{a.tokens_per_s_mean:.1f}~"   # short output: TTFT-dominated
            else:
                tps = f"{a.tokens_per_s_mean:.1f}"
            agg = f"{a.agg_tps_mean:.0f}" if getattr(a, "agg_tps_mean", None) else "—"
            table.add_row(
                a.endpoint, a.hardware, a.group,
                tps,
                agg,
                f"{a.ttft_ms_mean:.0f}" if a.ttft_ms_mean else "—",
                f"{a.ttft_ms_p95:.0f}" if a.ttft_ms_p95 else "—",
                f"{a.ttft_spike_rate*100:.0f}%" if a.ttft_spike_rate is not None else "—",
                f"{a.runaway_rate*100:.0f}%" if a.runaway_rate else "—",
                f"{a.quality_mean:.2f}" if a.quality_mean is not None else "—",
                f"{a.n_ok}/{a.n}",
            )
        _console.print(table)
    else:
        for a in by_cat:
            tps = f"{a.tokens_per_s_mean:.1f}" if a.tokens_per_s_mean else "—"
            out(f"{a.endpoint:20} {a.group:18} {tps:>8} tok/s  ok {a.n_ok}/{a.n}")


def _endpoint_preflight(cfg, allow_partial: bool = False) -> int:
    """Probe every endpoint; on any dead one, decide whether to proceed.

    Returns 0 to continue (cfg.endpoints is narrowed to the live set when some
    were dropped), or 1 to abort. All-dead always aborts.
    """
    import sys
    from .client import probe_endpoint

    async def _probe_all():
        results = []
        for ep in cfg.endpoints:
            ok, detail = await probe_endpoint(ep, timeout_s=min(cfg.timeout_s, 15))
            results.append((ep, ok, detail))
        return results

    out("Checking endpoints ...")
    probes = asyncio.run(_probe_all())
    live, dead = [], []
    for ep, ok, detail in probes:
        status = "OK" if ok else f"UNREACHABLE ({detail})"
        out(f"  {ep.name:16} {ep.model.split('/')[-1][:40]:40} {status}")
        (live if ok else dead).append(ep)

    if not dead:
        return 0
    if not live:
        out("\nAll endpoints are unreachable — nothing to benchmark. "
            "Start the inference server(s) and retry.")
        return 1

    dead_names = ", ".join(e.name for e in dead)
    live_names = ", ".join(e.name for e in live)
    out(f"\n{len(dead)} endpoint(s) unreachable: {dead_names}. "
        f"{len(live)} live: {live_names}.")

    proceed = allow_partial
    if not proceed and sys.stdin is not None and sys.stdin.isatty():
        try:
            ans = input(f"Proceed with only the live endpoint(s) [{live_names}]? [y/N] ")
        except EOFError:
            ans = ""
        proceed = ans.strip().lower() in ("y", "yes")

    if not proceed:
        out("Aborting. Re-run with --allow-partial to benchmark only the live "
            "endpoint(s), or bring the unreachable one(s) up.")
        return 1

    out(f"Proceeding with {len(live)} live endpoint(s): {live_names}.\n")
    cfg.endpoints = live
    return 0


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
    if args.concurrency is not None:
        cfg.concurrency = args.concurrency
    if args.parallel_endpoints is not None:
        cfg.parallel_endpoints = args.parallel_endpoints

    # Endpoint preflight: don't commit to a full benchmark against a dead
    # endpoint. Probe each one; if any are unreachable, only proceed on the
    # live ones with the user's explicit permission (--allow-partial, or an
    # interactive yes). This keeps a partial run an informed choice, not a
    # surprise, and makes single-endpoint runs work when a box is down.
    rc = _endpoint_preflight(cfg, allow_partial=args.allow_partial)
    if rc != 0:
        return rc

    # Fail fast on a misconfigured judge before running the whole benchmark.
    if cfg.judge and not args.no_judge:
        from .scoring.judge import preflight
        ok, msg = asyncio.run(preflight(cfg.judge, timeout_s=min(cfg.timeout_s, 60)))
        if not ok:
            out(f"Judge preflight FAILED: {msg}")
            out("Fix the judge model/key (e.g. OPENAI_API_KEY) and retry, "
                "or pass --no-judge to run without judging.")
            return 1
        out(f"Judge preflight: {msg}")

    out(f"Running benchmark: {len(cfg.endpoints)} endpoint(s), "
        f"tasks={cfg.tasks}, repeats={cfg.repeats}\n")

    output = asyncio.run(run(cfg, progress=out))

    if not args.no_score:
        from .scoring.run import score_results
        n = score_results(output.results)
        out(f"\nScored {n} results (reference-based quality checks).")

    if cfg.judge and not args.no_judge:
        from .scoring.judge import judge_results
        out(f"Judging subjective tasks with {cfg.judge.model} @ {cfg.judge.base_url} ...")
        nj = asyncio.run(judge_results(output.results, cfg.judge,
                                       timeout_s=cfg.timeout_s, concurrency=cfg.concurrency))
        out(f"Judged {nj} responses (LLM-judge, 1-5 rubric).")
    elif args.judge and not cfg.judge:
        out("--judge given but no `judge:` endpoint in config; skipping judging.")

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
    _print_quality_comparison(results)

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


def _cmd_leaderboard(args) -> int:
    """Aggregate multiple result JSONs into a cross-model leaderboard."""
    from .crossrun import load_runs, aggregate_by_model
    from .report import render_crossrun

    paths = args.results
    if not paths:
        paths = sorted(Path(args.dir).glob("*.json"))
        if not paths:
            out(f"No result JSONs found in {args.dir}/.")
            return 1
    loaded = load_runs(paths)
    if not loaded.results:
        out("No results loaded (unreadable or empty JSONs).")
        return 1
    rows, categories = aggregate_by_model(loaded.results)
    out(f"Pooled {len(loaded.results)} results from {len(loaded.run_meta)} run(s) "
        f"into {len(rows)} model/hardware unit(s).")
    for r in rows:
        out(f"  {r.model.split('/')[-1][:44]:44} @ {r.hardware:10} "
            f"{len(r.runs)} run(s), {r.samples} samples")

    out_html = Path(args.out) if args.out else Path(args.dir) / "leaderboard.html"
    render_crossrun(rows, categories, loaded.run_meta, out_html)
    out(f"\nLeaderboard -> {out_html}")
    if args.open:
        import webbrowser
        webbrowser.open(Path(out_html).resolve().as_uri())
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
    _print_quality_comparison(results)

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
                if ep.resolved_key and ep.resolved_key != "not-needed":
                    headers["Authorization"] = f"Bearer {ep.resolved_key}"
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
    r.add_argument("--concurrency", type=int,
                   help="Override concurrent requests per endpoint (1 = single-stream, "
                        "isolates true per-request latency).")
    r.add_argument("--parallel-endpoints", dest="parallel_endpoints",
                   action="store_true", default=None,
                   help="Benchmark all endpoints at once. Only for endpoints on "
                        "SEPARATE machines (shared-GPU endpoints would skew timings).")
    r.add_argument("--allow-partial", "--skip-unreachable", dest="allow_partial",
                   action="store_true",
                   help="If some endpoints are unreachable at preflight, proceed "
                        "with only the live ones instead of aborting (all-dead "
                        "still aborts). Grants permission up front for "
                        "non-interactive runs.")
    r.add_argument("--no-score", action="store_true",
                   help="Skip Phase 2 reference-based quality scoring.")
    r.add_argument("--judge", action="store_true",
                   help="Run the Phase 3 LLM-judge (requires a `judge:` endpoint in config).")
    r.add_argument("--no-judge", action="store_true",
                   help="Skip LLM-judging even if a judge endpoint is configured.")
    r.add_argument("--open", action="store_true", help="Open the HTML report when done.")
    r.set_defaults(func=_cmd_run)

    lt = sub.add_parser("list-tasks", help="List built-in task suites.")
    lt.set_defaults(func=_cmd_list_tasks)

    rp = sub.add_parser("report", help="Rebuild an HTML report from results JSON.")
    rp.add_argument("results")
    rp.add_argument("--out", "-o")
    rp.add_argument("--open", action="store_true")
    rp.set_defaults(func=_cmd_report)

    lb = sub.add_parser("leaderboard",
                        help="Aggregate multiple runs into a cross-model leaderboard.")
    lb.add_argument("results", nargs="*",
                    help="Result JSON files (default: all in --dir).")
    lb.add_argument("--dir", default="results",
                    help="Directory to scan for result JSONs when none are listed.")
    lb.add_argument("--out", "-o", help="Output HTML path (default: <dir>/leaderboard.html).")
    lb.add_argument("--open", action="store_true")
    lb.set_defaults(func=_cmd_leaderboard)

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


def _run_pyexec(script: str, extra: list[str]) -> int:
    """Self-hosted Python runner for frozen builds.

    In a PyInstaller exe, sys.executable is the aibench binary, not a Python
    interpreter, so the code-scoring check re-invokes this exe with a sentinel
    and we execute the candidate script here. The script reads its cases from
    sys.argv[1], so we shape argv to match a plain `python script.py <cases>`."""
    sys.argv = [script, *extra]
    with open(script, encoding="utf-8") as f:
        code = f.read()
    g = {"__name__": "__main__", "__file__": script}
    exec(compile(code, script, "exec"), g)  # noqa: S102 (sandboxed subprocess)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Frozen-exe self-exec hook: `aibench __pyexec__ <script> <args...>` runs the
    # script instead of the CLI (used by the Phase 2 code checks). Must run
    # before argparse. Only triggers on the exact sentinel, so normal CLI use is
    # unaffected.
    if argv is None and len(sys.argv) >= 3 and sys.argv[1] == "__pyexec__":
        return _run_pyexec(sys.argv[2], sys.argv[3:])
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
