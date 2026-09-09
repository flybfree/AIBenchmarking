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


def _headers(judge: Endpoint) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    key = judge.resolved_key
    if key and key != "not-needed":
        h["Authorization"] = f"Bearer {key}"
    return h


async def _adaptive_chat(
    client: httpx.AsyncClient, judge: Endpoint,
    messages: list[dict[str, str]], timeout_s: float, token_limit: int,
) -> httpx.Response:
    """POST a non-streaming chat completion, adapting to provider quirks:
    newer OpenAI models require `max_completion_tokens` (not `max_tokens`) and
    some reject a non-default `temperature`. Retries on those specific 400s so
    the same judge config works for local servers and hosted models alike.
    User-supplied `judge.params` always win (no auto token param is added if the
    user set one)."""
    use_completion_tokens = False
    include_temp = True
    r: httpx.Response | None = None
    for _ in range(4):
        payload: dict[str, Any] = {
            "model": judge.model, "messages": messages, "stream": False,
        }
        if not ({"max_tokens", "max_completion_tokens"} & set(judge.params)):
            key = "max_completion_tokens" if use_completion_tokens else "max_tokens"
            payload[key] = token_limit
        if include_temp and "temperature" not in judge.params:
            payload["temperature"] = 0.0
        payload.update(judge.params)

        r = await client.post(judge.chat_url, json=payload,
                              headers=_headers(judge), timeout=timeout_s)
        if r.status_code == 400:
            body = r.text.lower()
            if "max_completion_tokens" in body and not use_completion_tokens:
                use_completion_tokens = True
                continue
            if "temperature" in body and include_temp:
                include_temp = False
                continue
        return r
    return r  # type: ignore[return-value]


async def _judge_one(
    client: httpx.AsyncClient, judge: Endpoint, task: Task,
    result: RequestResult, timeout_s: float,
) -> tuple[float | None, str]:
    try:
        r = await _adaptive_chat(client, judge, _build_messages(task, result),
                                 timeout_s, token_limit=1024)
        if r.status_code != 200:
            return None, f"judge HTTP {r.status_code}: {r.text[:120].strip()}"
        content = r.json()["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
        return None, f"judge error: {type(e).__name__}: {e}"

    score, reason = parse_score(content)
    if score is None:
        return None, reason
    return (score - 1) / 4.0, f"{score}/5 - {reason}"


async def preflight(judge: Endpoint, timeout_s: float = 30.0) -> tuple[bool, str]:
    """Validate the judge endpoint (reachable, authorized, model exists, returns
    a parseable score) with one tiny call, so a misconfigured judge fails fast
    instead of after a full benchmark."""
    msgs = [{"role": "user",
             "content": 'Reply with only this JSON: {"score": 5, "reason": "ok"}'}]
    try:
        async with httpx.AsyncClient() as client:
            r = await _adaptive_chat(client, judge, msgs, timeout_s, token_limit=256)
    except httpx.HTTPError as e:
        return False, f"cannot reach judge: {type(e).__name__}: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}: {r.text[:200].strip()}"
    try:
        content = r.json()["choices"][0]["message"].get("content") or ""
    except (KeyError, ValueError, IndexError):
        return False, f"unexpected response shape: {r.text[:200].strip()}"
    score, _ = parse_score(content)
    if score is None:
        return False, f"judge did not return a parseable score: {content[:120]!r}"
    return True, f"OK (test score {score}/5)"


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
