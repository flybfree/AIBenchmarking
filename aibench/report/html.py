"""Self-contained HTML report generator.

Produces a single .html file with no external dependencies (inline CSS + inline
SVG bar charts), so it opens offline on any machine on the local network. Charts
group endpoints side by side within each task category for direct comparison.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from ..metrics import Aggregate

# A small categorical palette (color-blind-friendly-ish), assigned per endpoint.
_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759",
    "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
]


def _color_map(aggs: list[Aggregate]) -> dict[str, str]:
    endpoints = sorted({a.endpoint for a in aggs})
    return {e: _PALETTE[i % len(_PALETTE)] for i, e in enumerate(endpoints)}


def _grouped_bar_svg(
    aggs: list[Aggregate],
    value_fn,
    unit: str,
    colors: dict[str, str],
    higher_is_better: bool,
) -> str:
    """Render a grouped bar chart: groups = categories, bars = endpoints."""
    groups = sorted({a.group for a in aggs})
    endpoints = sorted({a.endpoint for a in aggs})
    lookup = {(a.endpoint, a.group): value_fn(a) for a in aggs}

    vals = [v for v in lookup.values() if v is not None]
    if not vals:
        return "<p class='muted'>No data.</p>"
    vmax = max(vals) * 1.15 or 1.0

    # Layout
    row_h = 26
    gap = 8
    group_gap = 22
    label_w = 150
    chart_w = 520
    bar_area = chart_w - label_w - 60
    y = 10
    rows: list[str] = []
    for g in groups:
        g_label = html.escape(g)
        rows.append(
            f"<text x='0' y='{y + row_h * 0.7}' class='glabel'>{g_label}</text>"
        )
        for e in endpoints:
            v = lookup.get((e, g))
            if v is None:
                w = 0
                vtxt = "n/a"
            else:
                w = max(2, (v / vmax) * bar_area)
                vtxt = f"{v:.1f} {unit}" if v >= 1 else f"{v:.2f} {unit}"
            rows.append(
                f"<rect x='{label_w}' y='{y}' width='{w:.1f}' height='{row_h - 6}' "
                f"rx='3' fill='{colors[e]}'><title>{html.escape(e)}: {vtxt}</title></rect>"
                f"<text x='{label_w + w + 6:.1f}' y='{y + row_h * 0.62}' "
                f"class='vlabel'>{vtxt}</text>"
            )
            y += row_h + gap
        y += group_gap
    height = y + 10
    legend = " ".join(
        f"<span class='chip'><i style='background:{colors[e]}'></i>{html.escape(e)}</span>"
        for e in endpoints
    )
    direction = "higher is better" if higher_is_better else "lower is better"
    return (
        f"<div class='legend'>{legend}<span class='muted'> — {direction}</span></div>"
        f"<svg viewBox='0 0 {chart_w} {height}' width='100%' "
        f"style='max-width:760px' role='img'>{''.join(rows)}</svg>"
    )


def _quality_class(q: float) -> str:
    if q >= 0.9:
        return "q-green"
    if q >= 0.6:
        return "q-amber"
    return "q-red"


def _quality_cell(a: Aggregate) -> str:
    if a.quality_mean is None:
        return "<span class='muted'>—</span>"
    return f"<b class='{_quality_class(a.quality_mean)}'>{a.quality_mean:.2f}</b>"


def _judge_cell(a: Aggregate) -> str:
    if a.judge_mean is None:
        return "<span class='muted'>—</span>"
    return f"<b class='{_quality_class(a.judge_mean)}'>{a.judge_mean:.2f}</b>"


def _table(aggs: list[Aggregate]) -> str:
    head = (
        "<tr><th>Endpoint</th><th>Hardware</th><th>Model</th><th>Task/Group</th>"
        "<th>tok/s (mean)</th><th>tok/s (median)</th><th>TTFT ms (mean)</th>"
        "<th>TTFT ms p95</th><th>spikes</th><th>runaway</th><th>total s</th><th>out tok</th>"
        "<th>quality</th><th>judge</th><th>ok/n</th><th>notes</th></tr>"
    )

    def spikes(a) -> str:
        if a.ttft_spike_rate is None:
            return "<span class='muted'>—</span>"
        pct = a.ttft_spike_rate * 100
        cls = "sflag" if a.ttft_spike_rate > 0.15 else "muted"
        return f"<span class='{cls}'>{pct:.0f}%</span>"

    def runaway(a) -> str:
        # fraction of runs that hit the token ceiling (finish=length) — an
        # unbounded-generation / thinking-runaway that spent the whole budget.
        if not getattr(a, "runaway_rate", None):
            return "<span class='muted'>—</span>"
        pct = a.runaway_rate * 100
        return f"<span class='sflag' title='hit token ceiling'>{pct:.0f}%</span>"

    def cell(v, fmt="{:.1f}"):
        return fmt.format(v) if v is not None else "—"

    def notes(a) -> str:
        flags = []
        if a.reasoning:
            flags.append("reasoning")
        if a.content_empty_n:
            flags.append(f"{a.content_empty_n} no-answer")
        if a.tps_fallback_n:
            flags.append(f"{a.tps_fallback_n} burst")
        return html.escape(", ".join(flags))

    def tps_cell(a) -> str:
        if a.tokens_per_s_mean is None:
            return "—"
        est = " *" if a.tokens_estimated else ""
        if a.tokens_per_s_std is not None:
            base = (f"{a.tokens_per_s_mean:.1f}"
                    f"<span class='pm'> &plusmn;{a.tokens_per_s_std:.1f}</span>")
        else:
            base = f"{a.tokens_per_s_mean:.1f}"
        # Outputs too short for tok/s to mean much (TTFT-dominated): grey it out
        # and mark it, so it doesn't read as a real throughput number.
        if getattr(a, "tps_unreliable", False):
            return (f"<span class='muted' title='outputs too short — "
                    f"TTFT-dominated'>&asymp;{base}{est}&nbsp;&dagger;</span>")
        return f"{base}{est}"

    rows = []
    for a in aggs:
        rows.append(
            "<tr>"
            f"<td>{html.escape(a.endpoint)}</td>"
            f"<td>{html.escape(a.hardware)}</td>"
            f"<td>{html.escape(a.model)}</td>"
            f"<td>{html.escape(a.group)}</td>"
            f"<td class='num'>{tps_cell(a)}</td>"
            f"<td class='num'>{cell(a.tokens_per_s_median)}</td>"
            f"<td class='num'>{cell(a.ttft_ms_mean)}</td>"
            f"<td class='num'>{cell(a.ttft_ms_p95)}</td>"
            f"<td class='num'>{spikes(a)}</td>"
            f"<td class='num'>{runaway(a)}</td>"
            f"<td class='num'>{cell(a.total_s_mean, '{:.2f}')}</td>"
            f"<td class='num'>{cell(a.completion_tokens_mean, '{:.0f}')}</td>"
            f"<td class='num'>{_quality_cell(a)}</td>"
            f"<td class='num'>{_judge_cell(a)}</td>"
            f"<td class='num'>{a.n_ok}/{a.n}</td>"
            f"<td class='muted'>{notes(a)}</td>"
            "</tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _sample_flags(r: Any) -> str:
    flags = []
    fr = getattr(r, "finish_reason", None)
    if fr and fr != "stop":
        flags.append(fr)
    if getattr(r, "content_empty", False):
        flags.append("no-answer")
    if getattr(r, "tps_from_total", False):
        flags.append("burst")
    return ", ".join(flags)


def _render_sample(r: Any, label: str) -> str:
    """Render one run's output: header line + tool calls / answer / reasoning."""
    text = getattr(r, "text", "") or ""
    reasoning = getattr(r, "reasoning_text", "") or ""
    tool_calls = getattr(r, "tool_calls", None) or []
    tps = getattr(r, "tokens_per_s", None)
    tps_txt = f"{tps:.1f} tok/s" if tps else ""
    flags = _sample_flags(r)
    flags_txt = f" · <span class='sflag'>{html.escape(flags)}</span>" if flags else ""

    quality = getattr(r, "quality", None)
    q_txt = ""
    if quality is not None:
        q_txt = f" · <b class='{_quality_class(quality)}'>quality {quality:.2f}</b>"
    judge = getattr(r, "judge_score", None)
    j_txt = ""
    if judge is not None:
        j_txt = f" · <b class='{_quality_class(judge)}'>judge {judge:.2f}</b>"
    note = getattr(r, "score_note", "") or ""
    note_txt = (
        f"<div class='scorenote muted'>{html.escape(note)}</div>" if note else ""
    )
    jnote = getattr(r, "judge_note", "") or ""
    jnote_txt = (
        f"<div class='scorenote judgenote'>&#9878; judge: {html.escape(jnote)}</div>"
        if jnote else ""
    )

    body_parts = []
    for tc in tool_calls:
        call = f"{tc.get('name', '')}({tc.get('arguments', '')})"
        body_parts.append(f"<div class='toolcall'>🔧 {html.escape(call)}</div>")
    if text.strip():
        body_parts.append(f"<pre class='answer'>{html.escape(text)}</pre>")
    elif not tool_calls:
        body_parts.append(
            "<p class='muted noans'>(no visible answer — spent the token "
            "budget reasoning; see thinking below)</p>"
        )
    if reasoning.strip():
        body_parts.append(
            "<details class='think'><summary>reasoning "
            f"({len(reasoning):,} chars)</summary>"
            f"<pre class='answer think'>{html.escape(reasoning)}</pre></details>"
        )

    return (
        f"<div class='sample'><div class='shead muted'>{html.escape(label)}"
        f" · {tps_txt}{q_txt}{j_txt}{flags_txt}</div>{note_txt}{jnote_txt}"
        f"{''.join(body_parts)}</div>"
    )


def _output_card(runs: list[Any], color: str) -> str:
    """One card per (endpoint, task) stacking every measured sample so
    consistency across repeats is visible at a glance."""
    head = runs[0]
    endpoint = html.escape(getattr(head, "endpoint", ""))
    hardware = html.escape(getattr(head, "hardware", ""))
    model = html.escape(getattr(head, "model", ""))
    n = len(runs)
    samples = "".join(
        _render_sample(r, f"sample {i}/{n}") for i, r in enumerate(runs, 1)
    )
    return (
        f"<div class='ocard' style='border-top:3px solid {color}'>"
        f"<div class='ohead'><b>{endpoint}</b> "
        f"<span class='muted'>· {hardware} · {model}</span>"
        f"<span class='otps'>{n} samples</span></div>"
        f"{samples}</div>"
    )


def _outputs_section(results: list[Any], colors: dict[str, str]) -> str:
    # Group runs by task, then by endpoint.
    by_task: dict[str, dict[str, list[Any]]] = {}
    task_category: dict[str, str] = {}
    task_prompt: dict[str, str] = {}
    for r in results:
        task = getattr(r, "task", "")
        ep = getattr(r, "endpoint", "")
        by_task.setdefault(task, {}).setdefault(ep, []).append(r)
        task_category.setdefault(task, getattr(r, "category", ""))
        if not task_prompt.get(task):
            task_prompt[task] = getattr(r, "prompt", "") or ""

    blocks = []
    for task in sorted(by_task, key=lambda t: (task_category.get(t, ""), t)):
        cat = html.escape(task_category.get(task, ""))
        prompt = html.escape(task_prompt.get(task, ""))
        cards = "".join(
            _output_card(runs, colors.get(ep, "#888"))
            for ep, runs in sorted(by_task[task].items())
        )
        blocks.append(
            f"<details class='otask' open><summary><b>{html.escape(task)}</b> "
            f"<span class='muted'>({cat})</span></summary>"
            f"<div class='prompt'><span class='muted'>prompt:</span> {prompt}</div>"
            f"<div class='ogrid'>{cards}</div></details>"
        )
    return "".join(blocks)


_CSS = """
/* Palette as tokens so every element gets an explicit, high-contrast color
   in both themes — nothing relies on color inheritance, which themed viewers
   can break (see the dark-mode table where inherited cell text washed out). */
:root {
  color-scheme: light dark;
  --bg:#fafafa; --fg:#181a1f; --muted:#55585f;
  --card:#ffffff; --border:#e5e5e5; --hair:#e8e8e8;
  --th-bg:#eceef1; --row-alt:#f5f6f8; --soft-bg:#eef0f3;
  --code-bg:#ffffff; --code-border:#e7e7e7;
  --think-fg:#4f545e; --think-bg:#f4f1ec;
  --tool-bg:#eef6ff; --tool-border:#cfe3fb;
  --green:#1f7a33; --amber:#8a6d00; --red:#b3271b;
}
@media (prefers-color-scheme: dark){ :root {
  --bg:#151517; --fg:#eef0f4; --muted:#adb1bb;
  --card:#1d1d20; --border:#37373c; --hair:#2c2c30;
  --th-bg:#26262b; --row-alt:#1b1b1e; --soft-bg:#26262a;
  --code-bg:#111113; --code-border:#333;
  --think-fg:#c2c6cf; --think-bg:#201d1a;
  --tool-bg:#16263a; --tool-border:#284a6e;
  --green:#63d47f; --amber:#e6b24a; --red:#f47163;
} }
body { font: 14px/1.5 system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--fg); }
.wrap { max-width: 900px; margin: 0 auto; padding: 24px; }
h1, h2 { color: var(--fg); }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 28px 0 10px; }
.muted { color: var(--muted); font-size: 12px; }
.card { background:var(--card); color:var(--fg); border:1px solid var(--border); border-radius:10px; padding:16px; margin:12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; overflow-x:auto; display:block; }
th, td { padding: 6px 9px; text-align: left; border-bottom: 1px solid var(--hair); white-space: nowrap; color: var(--fg); }
th { background:var(--th-bg); position: sticky; top:0; font-weight:600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:nth-child(even) td { background:var(--row-alt); }
.glabel { font-size: 12px; font-weight: 600; fill: var(--fg); }
.vlabel { font-size: 11px; fill: var(--muted); }
.legend { margin: 6px 0 10px; font-size: 12px; }
.chip { display:inline-flex; align-items:center; margin-right:12px; }
.chip i { width:11px; height:11px; border-radius:2px; display:inline-block; margin-right:5px; }
.otask { margin: 10px 0; border:1px solid var(--border); border-radius:10px; padding:6px 14px; background:var(--card); color:var(--fg); }
.otask > summary { cursor:pointer; font-size:15px; padding:6px 0; }
.prompt { font-size:12.5px; margin:6px 0 12px; padding:8px 10px; background:var(--soft-bg); border-radius:6px; }
.ogrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:12px; margin-bottom:8px; }
.ocard { border:1px solid var(--border); border-radius:8px; padding:10px 12px; background:var(--card); color:var(--fg); min-width:0; }
.ohead { font-size:12.5px; margin-bottom:8px; display:flex; flex-wrap:wrap; align-items:baseline; gap:6px; }
.otps { margin-left:auto; font-variant-numeric:tabular-nums; color:var(--muted); }
pre.answer { white-space:pre-wrap; word-break:break-word; font:12.5px/1.5 ui-monospace, "Cascadia Code", Consolas, monospace; margin:0; max-height:420px; overflow:auto; color:var(--fg); background:var(--code-bg); border:1px solid var(--code-border); border-radius:6px; padding:10px; }
pre.answer.think { color:var(--think-fg); background:var(--think-bg); }
.toolcall { font:12.5px ui-monospace, Consolas, monospace; color:var(--fg); background:var(--tool-bg); border:1px solid var(--tool-border); border-radius:6px; padding:8px 10px; margin-bottom:8px; }
.noans { font-style:italic; }
.sample { margin-bottom:12px; }
.sample + .sample { border-top:1px dashed var(--hair); padding-top:10px; }
.shead { font-size:11.5px; margin-bottom:6px; font-variant-numeric:tabular-nums; }
.sflag, .q-red { color:var(--red); }
.scorenote { font-size:11px; margin:-2px 0 8px; color:var(--muted); }
.judgenote { border-left:2px solid var(--amber); padding-left:6px; color:var(--fg); font-style:italic; }
.q-green { color:var(--green); }
.q-amber { color:var(--amber); }
.think > summary { cursor:pointer; font-size:12px; color:var(--muted); margin-top:8px; }
.mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size:12px; }
.pm { color:var(--muted); font-size:11px; }
.diag { border-left:4px solid var(--amber); }
.diaglist { margin:8px 0 0; padding-left:20px; }
.diaglist li { margin-bottom:8px; line-height:1.5; }
"""


def _scorecard_html(profiles: list[Any], categories: list[str], fits: list[Any]) -> str:
    if not profiles:
        return ""

    # Use-case fit recommendations.
    fit_rows = "".join(
        f"<tr><td><b>{html.escape(f.category)}</b></td>"
        f"<td><b>{html.escape(f.pick_model)}</b> "
        f"<span class='muted'>({html.escape(f.pick_endpoint)})</span></td>"
        f"<td class='muted'>{html.escape(f.reason)}</td></tr>"
        for f in fits
    )
    fit_table = (
        "<h3>Use-case fit — recommended model per use case</h3>"
        "<table><tr><th>Use case</th><th>Pick</th><th>Why</th></tr>"
        f"{fit_rows}</table>"
        "<p class='muted'>Picks the best-quality model, unless a faster one "
        "reaches comparable quality (within 0.05) — then the faster one wins. "
        "Quality is the LLM-judge score where available, else the objective checks.</p>"
    )

    # Model x use-case matrix: each cell = quality (colored) over tok/s.
    head = "<tr><th>Model</th><th>Hardware</th>" + "".join(
        f"<th>{html.escape(c)}</th>" for c in categories
    ) + "</tr>"
    rows = []
    for p in profiles:
        cells = []
        for c in categories:
            cell = p.cats.get(c)
            if not cell or cell.quality is None:
                cells.append("<td class='muted'>—</td>")
                continue
            qcls = _quality_class(cell.quality)
            tps = f"{cell.tps:.0f} tok/s" if cell.tps else "—"
            cells.append(
                f"<td class='num'><b class='{qcls}'>{cell.quality:.2f}</b>"
                f"<div class='pm'>{tps}</div></td>"
            )
        rows.append(
            f"<tr><td><b>{html.escape(p.model)}</b></td>"
            f"<td>{html.escape(p.hardware)}</td>{''.join(cells)}</tr>"
        )
    matrix = (
        "<h3>Model &times; use-case matrix — quality (0&ndash;1) over throughput</h3>"
        f"<table>{head}{''.join(rows)}</table>"
        "<p class='muted'>Quality uses the judge score for creative/summarization "
        "when judged, objective checks otherwise. Read across a row for a model's "
        "strengths/weaknesses; down a column to compare models for one use case.</p>"
    )

    return ("<h2>Scorecard — model strengths by use case</h2>"
            f"<div class='card'>{fit_table}</div>"
            f"<div class='card'>{matrix}</div>")


def _endpoints_html(endpoints: list[dict]) -> str:
    if not endpoints:
        return ""
    rows = []
    for e in endpoints:
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(str(e.get('name', '')))}</b></td>"
            f"<td>{html.escape(str(e.get('hardware', '') or '—'))}</td>"
            f"<td class='mono'>{html.escape(str(e.get('model', '')))}</td>"
            f"<td class='mono muted'>{html.escape(str(e.get('base_url', '')))}</td>"
            "</tr>"
        )
    head = ("<tr><th>Endpoint</th><th>Hardware</th><th>Model</th>"
            "<th>Base URL</th></tr>")
    models = {e.get("model", "") for e in endpoints}
    note = ""
    if len(models) > 1:
        note = (
            "<p class='muted'>Endpoints run <b>different models</b> — compare "
            "throughput/quality across them with that in mind (it is not a pure "
            "hardware A/B).</p>"
        )
    return (
        "<h2>Endpoints</h2>"
        f"<div class='card'><table>{head}{''.join(rows)}</table>{note}</div>"
    )


def _quality_comparison_html(comparisons: list[Any]) -> str:
    if not comparisons:
        return ""
    rows = []
    for c in comparisons:
        if c.significant:
            verdict = "<b class='q-green'>significant</b>"
        elif c.z == 0.0 and abs(c.leader_mean - c.other_mean) < 0.03:
            verdict = "<b class='muted'>tied</b>"
        elif c.z is not None:
            verdict = "<b class='q-amber'>within noise</b>"
        else:
            verdict = "<b class='muted'>need repeats</b>"
        zt = ("—" if c.z is None else ("∞" if c.z == float("inf") else f"{c.z:.1f}"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(c.category)}</td>"
            f"<td><b>{html.escape(c.leader)}</b> {c.leader_mean:.2f}</td>"
            f"<td>{html.escape(c.other)} {c.other_mean:.2f}</td>"
            f"<td class='num'>{zt}</td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    head = ("<tr><th>Category</th><th>Higher quality</th><th>vs</th>"
            "<th>Δ/SE</th><th>verdict</th></tr>")
    return (
        "<h2>Head-to-head — quality significance</h2>"
        "<div class='card'><p class='muted'>Compares the two highest-quality "
        "endpoints per category (judge score where available, else objective "
        "checks). A quality gap between the same model on different hardware is "
        "expected to be noise — <b>within noise</b> confirms that. Δ/SE&ge;2 "
        "means the gap is unlikely to be sampling variance.</p>"
        f"<table>{head}{''.join(rows)}</table></div>"
    )


def _comparison_html(comparisons: list[Any], variance: list[Any]) -> str:
    if not comparisons:
        return ""
    rows = []
    for c in comparisons:
        if getattr(c, "unreliable", False):
            verdict = "<b class='muted' title='outputs too short — TTFT-dominated'>n/a — outputs too short</b>"
        elif c.significant:
            verdict = "<b class='q-green'>significant</b>"
        elif c.z is not None:
            verdict = "<b class='q-amber'>within noise</b>"
        else:
            verdict = "<b class='muted'>need repeats</b>"
        pct_cell = ("<span class='muted'>&asymp;+%.0f%%</span>" % c.pct
                    if getattr(c, "unreliable", False) else "+%.0f%%" % c.pct)
        rows.append(
            "<tr>"
            f"<td>{html.escape(c.category)}</td>"
            f"<td><b>{html.escape(c.faster)}</b> {c.faster_mean:.1f}</td>"
            f"<td>{html.escape(c.slower)} {c.slower_mean:.1f}</td>"
            f"<td class='num'>{pct_cell}</td>"
            f"<td class='num'>{'%.1f' % c.z if c.z is not None else '—'}</td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    head = ("<tr><th>Category</th><th>Faster (tok/s)</th><th>vs (tok/s)</th>"
            "<th>Δ</th><th>Δ/SE</th><th>verdict</th></tr>")
    var_note = ""
    if variance:
        items = "".join(
            f"<li>{html.escape(v.endpoint)} / {html.escape(v.category)}: "
            f"{v.mean:.1f} &plusmn;{v.std:.1f} tok/s "
            f"(CV {v.cv*100:.0f}%) — noisy; add repeats.</li>"
            for v in variance
        )
        var_note = (
            "<p class='muted'>High run-to-run variance (throughput swings a lot "
            "between repeats), so treat these as soft:</p>"
            f"<ul class='diaglist'>{items}</ul>"
        )
    return (
        "<h2>Head-to-head — throughput significance</h2>"
        "<div class='card'><p class='muted'>Compares the two fastest endpoints "
        "per category. <b>Δ/SE</b> is the gap divided by its standard error; "
        "&ge;2 means the difference is unlikely to be noise. With few repeats "
        "this is approximate.</p>"
        f"<table>{head}{''.join(rows)}</table>{var_note}</div>"
    )


def _diagnostics_html(diagnostics: list[Any]) -> str:
    if not diagnostics:
        return ""
    icon = {"offload": "🐌", "ttft": "⏱️"}
    items = "".join(
        f"<li><b>{icon.get(d.kind, '⚠️')} {html.escape(d.endpoint)}</b> "
        f"({html.escape(d.kind)}): {html.escape(d.message)}</li>"
        for d in diagnostics
    )
    return (
        "<h2>Diagnostics — worth a look</h2>"
        "<div class='card diag'><p class='muted'>These patterns compare "
        "endpoints running the <b>same model</b> and usually indicate a setup "
        "issue rather than a real hardware difference.</p>"
        f"<ul class='diaglist'>{items}</ul></div>"
    )


def render(
    by_task: list[Aggregate],
    by_category: list[Aggregate],
    meta: dict,
    out_path: str | Path,
    results: list[Any] | None = None,
    diagnostics: list[Any] | None = None,
    endpoints: list[dict] | None = None,
    comparisons: list[Any] | None = None,
    variance: list[Any] | None = None,
    profiles: list[Any] | None = None,
    categories: list[str] | None = None,
    fits: list[Any] | None = None,
    quality_comparisons: list[Any] | None = None,
) -> Path:
    colors = _color_map(by_category or by_task)
    tps_chart = _grouped_bar_svg(
        by_category, lambda a: a.tokens_per_s_mean, "tok/s", colors, True
    )
    ttft_chart = _grouped_bar_svg(
        by_category, lambda a: a.ttft_ms_mean, "ms", colors, False
    )
    has_quality = any(a.quality_mean is not None for a in by_category)
    quality_chart = ""
    if has_quality:
        quality_chart = (
            "<h2>Quality by category — reference score (0–100%)</h2>"
            "<div class='card'>"
            + _grouped_bar_svg(
                [a for a in by_category if a.quality_mean is not None],
                lambda a: (a.quality_mean or 0) * 100, "%", colors, True,
            )
            + "</div>"
        )

    meta_rows = "".join(
        f"<tr><td class='muted'>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in meta.items()
    )
    note_bits = []
    if any((a.ttft_spike_rate or 0) > 0 for a in by_task):
        note_bits.append(
            "<b>spikes</b>: fraction of runs whose time-to-first-token was an "
            "outlier (&gt;3x the median and &gt;250 ms) — usually inference-server "
            "jitter, not steady-state latency. Read TTFT as the median/mean; a "
            "high spike rate means the mean is inflated by a few slow first tokens."
        )
    if any((a.runaway_rate or 0) > 0 for a in by_task):
        note_bits.append(
            "<b>runaway</b>: fraction of runs that hit the token ceiling "
            "(finish=length) — the model spent its whole budget generating "
            "(usually unbounded thinking) and was cut off. A persistent rate "
            "that survives a larger budget is a model stability issue, not a "
            "cap that's merely too low."
        )
    if any(getattr(a, "tps_unreliable", False) for a in by_task):
        note_bits.append(
            "<b>&asymp; &dagger; tok/s</b>: outputs in this group are too short "
            "(median &lt;150 tokens) for tokens/sec to be meaningful — the number "
            "is dominated by time-to-first-token, not decode speed. Compare TTFT "
            "for these instead; they're excluded from throughput significance."
        )
    if any(a.tokens_estimated for a in by_task):
        note_bits.append(
            "* token count estimated from streamed chunks (server did not "
            "report usage)."
        )
    if any(a.reasoning for a in by_task):
        note_bits.append(
            "<b>reasoning</b>: model emits hidden thinking tokens — TTFT is time "
            "to the first such token and tok/s covers the whole generation."
        )
    if any(a.content_empty_n for a in by_task):
        note_bits.append(
            "<b>no-answer</b>: run produced only reasoning and no visible answer "
            "(usually hit the token cap while thinking — raise max_tokens for a "
            "task-completion comparison)."
        )
    if any(a.tps_fallback_n for a in by_task):
        note_bits.append(
            "<b>burst</b>: whole generation arrived in one batch, so tok/s is the "
            "end-to-end rate (completion tokens / total time), not per-token decode."
        )
    note = (
        "<p class='muted'>" + "<br>".join(note_bits) + "</p>" if note_bits else ""
    )

    outputs_html = ""
    if results:
        outputs_html = (
            "<h2>Outputs — read the responses to judge quality</h2>"
            "<p class='muted'>Every measured sample per model per task, stacked "
            "so you can gauge consistency across repeats. Hidden reasoning is "
            "collapsed; per-sample flags (e.g. length, no-answer) are shown in red.</p>"
            + _outputs_section(results, colors)
        )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aibench report</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>AI Benchmark Report</h1>
<div class="card"><table>{meta_rows}</table></div>

{_endpoints_html(endpoints or [])}

{_scorecard_html(profiles or [], categories or [], fits or [])}

{_diagnostics_html(diagnostics or [])}

<h2>Throughput by category — tokens/sec</h2>
<div class="card">{tps_chart}</div>

<h2>Latency by category — time to first token</h2>
<div class="card">{ttft_chart}</div>

{quality_chart}

<h2>Per-category summary</h2>
<div class="card">{_table(by_category)}</div>

{_comparison_html(comparisons or [], variance or [])}

{_quality_comparison_html(quality_comparisons or [])}

<h2>Per-task detail</h2>
<div class="card">{_table(by_task)}{note}</div>

{outputs_html}

<p class="muted">Generated by aibench. Performance metrics + Phase 2 reference-based
quality scoring. Creative-writing quality is compliance-only (length/structure);
subjective quality is a future LLM-judge phase.</p>
</div></body></html>"""

    p = Path(out_path)
    p.write_text(doc, encoding="utf-8")
    return p


def render_crossrun(rows: list[Any], categories: list[str],
                    run_meta: list[dict], out_path: str | Path) -> Path:
    """Cross-run leaderboard: every (model @ hardware) pooled across runs."""
    # Runs included.
    run_rows = "".join(
        f"<tr><td>{html.escape(m['label'])}</td>"
        f"<td class='muted'>{html.escape(m['started'])}</td>"
        f"<td class='mono'>{html.escape(', '.join(mm.split('/')[-1][:40] for mm in m['models']))}</td>"
        f"</tr>"
        for m in run_meta
    )
    runs_table = (
        "<h2>Runs included</h2><div class='card'><table>"
        "<tr><th>Run</th><th>Started</th><th>Models</th></tr>"
        f"{run_rows}</table></div>"
    )

    # Per-use-case leaders (ranked by effective quality, then throughput).
    lead_blocks = []
    for cat in categories:
        ranked = sorted(
            [r for r in rows if cat in r.cats and r.cats[cat].quality is not None
             and not getattr(r.cats[cat], "failed", False)],
            key=lambda r: (r.cats[cat].quality, r.cats[cat].tps or 0),
            reverse=True,
        )
        if not ranked:
            continue
        items = "".join(
            f"<li><b class='{_quality_class(r.cats[cat].quality)}'>{r.cats[cat].quality:.2f}</b> "
            f"{html.escape(r.model.split('/')[-1][:44])} "
            f"<span class='muted'>@ {html.escape(r.hardware)} · "
            f"{r.cats[cat].tps:.0f} tok/s</span></li>"
            if r.cats[cat].tps else
            f"<li><b class='{_quality_class(r.cats[cat].quality)}'>{r.cats[cat].quality:.2f}</b> "
            f"{html.escape(r.model.split('/')[-1][:44])}</li>"
            for r in ranked
        )
        lead_blocks.append(
            f"<div class='leadcat'><h3>{html.escape(cat)}</h3><ol>{items}</ol></div>"
        )
    leaders = (
        "<h2>Leaders by use case — best quality first</h2>"
        f"<div class='card'><div class='leadgrid'>{''.join(lead_blocks)}</div>"
        "<p class='muted'>Quality is the LLM-judge score where a run judged that "
        "task, else the objective checks. Throughput is pooled across runs "
        "(noisier — different concurrency/load).</p></div>"
    )

    # Full model x use-case matrix, ranked best-first by overall quality.
    head = ("<tr><th>Model</th><th>Hardware</th><th>overall</th><th>runs</th>"
            + "".join(f"<th>{html.escape(c)}</th>" for c in categories) + "</tr>")
    mrows = []
    for r in rows:   # already ordered by overall quality from aggregate_by_model
        cells = []
        for c in categories:
            cell = r.cats.get(c)
            if getattr(cell, "failed", False):
                # Attempted but every request errored — a real failure, shown
                # distinctly from a never-tested "—" so it can't hide.
                cells.append(
                    f"<td class='num'><b class='q-red' title='all {cell.failed_n} "
                    f"request(s) errored'>FAIL</b><div class='pm'>{cell.failed_n} err</div></td>"
                )
                continue
            if not cell or cell.quality is None:
                cells.append("<td class='muted'>—</td>")
                continue
            tps = f"{cell.tps:.0f} tok/s" if cell.tps else ""
            tag = "" if cell.basis == "judged" else "<span class='pm'> (checks)</span>"
            exc = getattr(cell, "quality_excluded_runs", 0)
            broke = f"<span class='sflag' title='broken/truncated run(s) dropped'> &dagger;{exc}</span>" if exc else ""
            cells.append(
                f"<td class='num'><b class='{_quality_class(cell.quality)}'>"
                f"{cell.quality:.2f}</b>{broke}{tag}<div class='pm'>{tps} · n={cell.n}</div></td>"
            )
        # Overall quality (failed categories counted as 0) + completeness badge.
        oq = getattr(r, "overall_quality", None)
        if oq is None:
            overall_cell = "<td class='muted'>—</td>"
        else:
            done = f"{r.cats_ok}/{r.cats_attempted}"
            incomplete = r.cats_ok < r.cats_attempted
            badge = (f"<span class='sflag' title='{r.cats_attempted - r.cats_ok} "
                     f"category(ies) failed'> {done}</span>" if incomplete
                     else f"<span class='pm'> {done}</span>")
            overall_cell = (f"<td class='num'><b class='{_quality_class(oq)}'>"
                            f"{oq:.2f}</b>{badge}</td>")
        off = getattr(r, "offloaded_runs", 0)
        runs_cell = (f"{len(r.runs)}"
                     + (f"<span class='sflag'> &minus;{off} off</span>" if off else ""))
        mrows.append(
            f"<tr><td><b>{html.escape(r.model.split('/')[-1][:48])}</b></td>"
            f"<td>{html.escape(r.hardware)}</td>"
            f"{overall_cell}"
            f"<td class='num'>{runs_cell}</td>{''.join(cells)}</tr>"
        )
    any_off = any(getattr(r, "offloaded_runs", 0) for r in rows)
    any_broke = any(getattr(c, "quality_excluded_runs", 0)
                    for r in rows for c in r.cats.values())
    off_note = (
        "<br>&minus;N off = N run(s) where this unit was CPU-offloaded, excluded "
        "from the pooled throughput (quality still uses every run)."
        if any_off else ""
    )
    broke_note = (
        "<br>&dagger;N on a cell = N run(s) dropped from that quality because the "
        "output was broken/truncated (empty responses, not a genuine overflow) — "
        "a config/format bug rather than a real quality result."
        if any_broke else ""
    )
    matrix = (
        "<h2>Model &times; use-case matrix</h2>"
        f"<div class='card'><table>{head}{''.join(mrows)}</table>"
        "<p class='muted'>Ranked best-first by <b>overall</b> — the mean quality "
        "across the categories a model was actually run on, with a <b>FAIL</b>ed "
        "category counted as 0 so a model that can't do a use case can't outrank "
        "one that can. The badge beside it (e.g. 5/6) is how many attempted "
        "categories succeeded; a never-tested category is simply absent (—) and "
        "doesn't affect the score. <b>FAIL</b> = the category was attempted but "
        "every request errored (a serving/capability failure), distinct from — "
        "(never tested). Each quality cell: effective quality (judge where "
        "available, else objective '(checks)') over pooled throughput and sample "
        f"count.{off_note}{broke_note}</p></div>"
    )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aibench leaderboard</title><style>{_CSS}
.leadgrid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap:16px; }}
.leadcat h3 {{ margin:4px 0 6px; font-size:14px; }}
.leadcat ol {{ margin:0; padding-left:22px; }}
.leadcat li {{ margin-bottom:4px; }}
</style></head><body><div class="wrap">
<h1>AI Benchmark — Cross-run Leaderboard</h1>
<p class="muted">{len(rows)} model/hardware units pooled across {len(run_meta)} run(s).</p>

{leaders}

{matrix}

{runs_table}

<p class="muted">Generated by aibench. Pools multiple runs by (model @ hardware).
Quality pools cleanly across runs; throughput is noisier (varies with
concurrency/load per run) — use the sample count (n) as a confidence cue.</p>
</div></body></html>"""
    p = Path(out_path)
    p.write_text(doc, encoding="utf-8")
    return p
