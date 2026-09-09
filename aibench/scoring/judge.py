"""Phase 3: LLM-as-judge for graduated quality on subjective tasks.

Objective checks (code execution, format, tool correctness) answer "did it
work"; they can't tell a great story from a mediocre one. The judge scores the
subjective use cases (creative writing, summary quality) on a 1-5 rubric, so
models actually separate on quality.

The judge is any OpenAI-compatible endpoint (configured as `judge:` in the run
config) — typically a strong local model. Only tasks that define a `rubric` are
judged. Scores are normalised to [0, 1] (1->0.0, 3->0.5, 5->1.0) and stored on
`RequestResult.judge_score` alongside the objective `quality`.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from ..client import RequestResult
from ..config import Endpoint
from ..tasks import ALL_SUITES, Task

_SYSTEM = (
    "You are a strict, fair evaluator of AI model outputs. You are given a task "
    "and a candidate response. Score the response from 1 (poor) to 5 (excellent) "
    "using the given rubric. Be discerning: reserve 5 for genuinely excellent "
    "work and use the full range. Respond with ONLY a JSON object: "
    '{"score": <1-5>, "reason": "<one sentence>"}.'
)


def _tasks_by_id() -> dict[str, Task]:
    return {t.id: t for s in ALL_SUITES.values() for t in s.tasks}


def _build_messages(task: Task, result: RequestResult) -> list[dict[str, str]]:
    answer = result.text.strip() or "(the model produced no visible answer)"
    user = (
        f"# Task given to the model\n{task.prompt}\n\n"
        f"# Rubric\n{task.rubric}\n\n"
        f"# Candidate response\n{answer}\n\n"
        "Score it 1-5 per the rubric. Reply with only the JSON object."
    )
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user}]


def parse_score(text: str) -> tuple[int | None, str]:
    """Extract a 1-5 score and reason from the judge's reply, tolerating extra
    prose or reasoning around the JSON."""
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            s = int(round(float(obj["score"])))
            if 1 <= s <= 5:
                return s, str(obj.get("reason", "")).strip()
        except (ValueError, KeyError, TypeError):
            pass
    m = re.search(r"score\D{0,4}([1-5])(?:\s*/\s*5)?", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), text.strip()[:120]
    m = re.search(r"\b([1-5])\s*/\s*5\b", text)
    if m:
        return int(m.group(1)), text.strip()[:120]
    return None, f"could not parse score from: {text.strip()[:80]!r}"


async def _judge_one(
    client: httpx.AsyncClient, judge: Endpoint, task: Task,
    result: RequestResult, timeout_s: float,
) -> tuple[float | None, str]:
    payload = {
        "model": judge.model,
        "messages": _build_messages(task, result),
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 1024,   # headroom in case the judge is a reasoning model
    }
    payload.update(judge.params)
    headers = {"Content-Type": "application/json"}
    key = judge.resolved_key
    if key and key != "not-needed":
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = await client.post(judge.chat_url, json=payload, headers=headers,
                              timeout=timeout_s)
        if r.status_code != 200:
            return None, f"judge HTTP {r.status_code}"
        content = r.json()["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
        return None, f"judge error: {type(e).__name__}: {e}"

    score, reason = parse_score(content)
    if score is None:
        return None, reason
    return (score - 1) / 4.0, f"{score}/5 - {reason}"


async def judge_results(
    results: list[RequestResult], judge: Endpoint,
    timeout_s: float = 120.0, concurrency: int = 4,
) -> int:
    """Judge every result whose task defines a rubric. Returns count judged."""
    tasks = _tasks_by_id()
    targets = [
        r for r in results
        if r.ok and (t := tasks.get(r.task)) is not None and t.rubric
    ]
    if not targets:
        return 0

    sem = asyncio.Semaphore(max(1, concurrency))
    limits = httpx.Limits(max_connections=max(2, concurrency + 1))

    async with httpx.AsyncClient(limits=limits) as client:
        async def go(r: RequestResult) -> None:
            async with sem:
                score, note = await _judge_one(
                    client, judge, tasks[r.task], r, timeout_s
                )
            r.judge_score = score
            r.judge_note = note

        await asyncio.gather(*(go(r) for r in targets))
    return sum(1 for r in targets if r.judge_score is not None)
