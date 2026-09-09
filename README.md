# aibench — Local AI Inference Benchmarking

Send inference requests to AI endpoints on your local network and compare the
**performance of local models and hardware** across task categories: creative
writing, code generation, summarization, and agents/tool use.

Works with any **OpenAI-compatible** server: **Ollama**, **LM Studio**,
**vLLM**, **llama.cpp** server, **TGI**, text-generation-webui, and others.

## Status: Phase 1 — performance metrics

Scoring is intentionally phased (the architecture is pluggable):

| Phase | What it measures | Status |
|-------|------------------|--------|
| **1** | Performance: tokens/sec, time-to-first-token (TTFT), total latency, throughput | ✅ |
| **2** | Reference-based quality: code execution, SQL execution, format & keyword checks, tool-call correctness | ✅ |
| **3** | LLM-as-judge for subjective tasks (creative writing) | optional |

### Phase 2 quality scoring

Each task carries deterministic, offline `checks` (see
[aibench/scoring/checks.py](aibench/scoring/checks.py)) that produce a quality
score in `[0, 1]`:

The suite spans **easy→hard** tasks per category (20 total, `list-tasks` shows
them) chosen to *discriminate* — weaker models fail the hard ones, so scores
don't all pin at 1.0. Check types (see [checks.py](aibench/scoring/checks.py)):

| Category | Example checks | How |
|------|-------|-----|
| code (fib, valid-parens, roman, merge-intervals, edit-distance) | `python_func` | Runs the generated function in an isolated subprocess against hidden edge cases (pass rate) |
| code (SQL) | `sql_sqlite` | Executes the query on a seeded in-memory SQLite DB; checks the top-5 order (catches a missing year filter) |
| summarization | `bullets`, `sentence_count`, `keyword_coverage` | Format compliance + key-concept coverage (3-bullet, 1-sentence TL;DR, action-item extraction) |
| tool use | `tool_call` | Correct tool, argument accuracy, tool *selection* among several, and *restraint* (not calling when unneeded) |
| agents (multi-turn) | `tool_sequence` + `keyword_coverage` | Real tool loop — the harness executes tools between turns; scored on whether the model chained the right tools and reached the correct answer (see [agent.py](aibench/agent.py)) |
| creative writing | `word_count`, `stanza_count`, `acrostic` | Objectively-checkable constraints (exact six-word story, acrostic spelling); subjective quality is Phase 3 |

**Multi-turn agents:** the `agents` suite gives the model tools with real
implementations (`tool_impls`). When the model calls a tool, the harness runs it,
feeds the result back, and continues — up to `max_turns` — testing tool
selection, chaining one tool's output into the next, and synthesising a final
answer. Tasks include a weather comparison, a chained customer→order-count
lookup, and a stock-value calculation.

### Trusting the numbers: variance & significance

Reports show throughput as **mean ±std**, and a **Head-to-head** table compares
the two fastest endpoints per category with a significance estimate
(`Δ/SE` = gap ÷ standard error; ≥2 ≈ unlikely to be noise). Endpoints with
noisy throughput (high coefficient of variation) are flagged with a
"add repeats" note. See [compare.py](aibench/compare.py).

Scoring runs automatically after `run` (disable with `--no-score`), and can be
applied to any past results JSON without re-benchmarking:

```bash
python -m aibench score results/bakeoff_20260908_215646.json --open
```

> Note: code checks execute model-generated code (subprocess + timeout
> isolation). Run against models and tasks you trust.

## Install

```bash
python -m pip install -r requirements.txt
```

## Quick start

1. Write a config (endpoints on your network):

   ```bash
   python -m aibench init-config configs/example.yaml
   ```

   Then edit `configs/example.yaml` — set each endpoint's `base_url`, `model`,
   and a `hardware` label (used to compare the same model across machines):

   ```yaml
   label: "bakeoff"
   repeats: 3          # measured runs per (endpoint, task)
   warmup: 1           # discarded warmup runs (model load / cache)
   concurrency: 1      # >1 measures throughput under concurrent load
   tasks:
     - all             # or: creative_writing, code_generation, summarization, tool_use
   endpoints:
     - name: "rig1-llama3-8b"
       base_url: "http://192.168.1.50:11434/v1"   # Ollama
       model: "llama3:8b"
       hardware: "RTX 4090"
     - name: "rig2-qwen2.5-7b"
       base_url: "http://192.168.1.51:1234/v1"     # LM Studio
       model: "qwen2.5-7b-instruct"
       hardware: "M2 Max"
   ```

2. Check connectivity:

   ```bash
   python -m aibench ping --config configs/example.yaml
   ```

3. Run the benchmark (saves JSON + a self-contained HTML report):

   ```bash
   python -m aibench run --config configs/example.yaml --open
   ```

## Commands

| Command | Purpose |
|---------|---------|
| `run --config <f> [--open] [--no-score] [--max-tokens N] [--reasoning-reserve N] [--repeats N]` | Run the benchmark, score quality, build an HTML report. |
| `score <results.json> [--open]` | (Re)score a saved run and rebuild its report — no re-benchmarking. |
| `ping --config <f>` | Quick connectivity/latency check of each endpoint. |
| `list-tasks` | Show built-in task suites and their prompts' token budgets. |
| `report <results.json> [--open]` | Rebuild the HTML report from saved results. |
| `init-config [path]` | Write a starter config you can edit. |

## What gets measured (Phase 1)

Per request, streaming so timings are precise:

- **TTFT** — time to first content token (prompt-processing + queue latency).
- **Total latency** — request start to stream end.
- **tokens/sec** — completion tokens ÷ generation time (first token → last).
- **Output tokens** — from the server's `usage` field when provided; otherwise
  estimated from streamed chunks and flagged with `*` in reports.

Results are aggregated per task and per category, with mean/median tok/s and
mean/p95 TTFT over `repeats` runs. Warmup runs are discarded.

Raw per-run data is saved as JSON in `results/`, so Phase 2/3 scorers can be
run over past results without re-benchmarking.

## How it's organized

```
aibench/
  config.py      # YAML/JSON run config -> Endpoint / RunConfig
  client.py      # streaming OpenAI-compatible client + timing capture
  tasks/         # built-in task suites (base.py defines Task/TaskSuite)
  scoring/       # pluggable scorers: reference.py (Phase 2), checks.py, extract.py
  runner.py      # endpoint x task matrix, warmup + repeats + concurrency
  metrics.py     # aggregation into comparable summary stats
  report/        # self-contained HTML report (inline SVG charts, offline)
  cli.py         # command-line interface
```

## Diagnostics

When two endpoints run the **same model**, the report and terminal auto-flag
patterns that usually mean a setup problem rather than a real hardware
difference (see [aibench/diagnose.py](aibench/diagnose.py)):

- **offload** — one endpoint's throughput is under ~40% of a same-model peer's,
  which usually means the model didn't fit in VRAM and spilled to CPU (or a
  heavier quant/config is loaded).
- **ttft** — one endpoint's time-to-first-token is ≥1.8× a peer's without a
  matching throughput gain, which points to a server-config difference
  (request batching, context length) rather than hardware.

These are tuned so an ordinary GPU-vs-GPU gap (a ~1.5–2× bandwidth difference)
does not trip them.

## Notes

- Endpoints are benchmarked **sequentially** so they don't compete for the same
  hardware and skew timings; requests **within** an endpoint honor `concurrency`.
- The HTML report is fully self-contained (inline CSS + SVG), so it opens
  offline on any machine on the network.
