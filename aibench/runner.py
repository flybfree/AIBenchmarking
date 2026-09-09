"""Benchmark orchestration.

For each endpoint, run each task `warmup + repeats` times (warmup runs are
discarded so model-load and cache effects don't skew the numbers). Requests to a
single endpoint are issued with bounded concurrency; endpoints are processed
sequentially so they don't compete for the same hardware and distort timing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import httpx

from .client import RequestResult, run_request
from .config import Endpoint, RunConfig
from .tasks import Task, resolve_suites

ProgressCb = Callable[[str], None]


@dataclass
class RunOutput:
    config: RunConfig
    started_at: str
    finished_at: str = ""
    results: list[RequestResult] = field(default_factory=list)


def _collect_tasks(suite_names: list[str]) -> list[Task]:
    tasks: list[Task] = []
    for suite in resolve_suites(suite_names):
        tasks.extend(suite.tasks)
    return tasks


def _merge_params(cfg: RunConfig, task: Task) -> dict:
    """Effective output budget = answer allowance + reasoning reserve.

    A task's `max_tokens` (or the global `--max-tokens` override) is the room
    for the visible answer; `reasoning_reserve` is added on top so reasoning
    models can think without eating into the answer. If neither an answer
    budget nor a global override is set, no cap is sent (server default)."""
    params = dict(task.params)
    base = cfg.max_tokens if cfg.max_tokens is not None else params.get("max_tokens")
    if base is not None:
        params["max_tokens"] = base + max(0, cfg.reasoning_reserve)
    return params


async def _run_endpoint(
    cfg: RunConfig,
    endpoint: Endpoint,
    tasks: list[Task],
    progress: ProgressCb,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    limits = httpx.Limits(max_connections=max(2, cfg.concurrency + 1))

    async with httpx.AsyncClient(limits=limits) as client:
        for task in tasks:
            params = _merge_params(cfg, task)

            async def one(run_idx: int, is_warmup: bool) -> None:
                async with sem:
                    res = await run_request(
                        client,
                        endpoint,
                        task.id,
                        task.messages(),
                        params,
                        cfg.timeout_s,
                        category=task.category,
                        tools=task.tools,
                    )
                tag = "warmup" if is_warmup else f"run {run_idx}"
                if res.ok:
                    tps = f"{res.tokens_per_s:.1f} tok/s" if res.tokens_per_s else "ok"
                    progress(f"  [{endpoint.name}] {task.id} ({tag}): {tps}")
                else:
                    progress(f"  [{endpoint.name}] {task.id} ({tag}): ERROR {res.error}")
                if not is_warmup:
                    results.append(res)

            # Warmup runs (discarded).
            for w in range(cfg.warmup):
                await one(w, True)
            # Measured runs — respect concurrency for throughput-under-load.
            if cfg.concurrency > 1:
                await asyncio.gather(
                    *(one(i, False) for i in range(cfg.repeats))
                )
            else:
                for i in range(cfg.repeats):
                    await one(i, False)

    return results


async def run(cfg: RunConfig, progress: ProgressCb = print) -> RunOutput:
    tasks = _collect_tasks(cfg.tasks)
    if not tasks:
        raise ValueError("No tasks resolved from config.")
    out = RunOutput(
        config=cfg,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    for endpoint in cfg.endpoints:
        progress(f"[endpoint] {endpoint.name} ({endpoint.model} @ {endpoint.base_url})")
        res = await _run_endpoint(cfg, endpoint, tasks, progress)
        out.results.extend(res)
    out.finished_at = datetime.now(timezone.utc).isoformat()
    return out
