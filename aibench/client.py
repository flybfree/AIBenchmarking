"""OpenAI-compatible streaming client with precise timing capture.

We stream responses so we can measure time-to-first-token (TTFT) separately from
total generation time. Token counts come from the server's `usage` field when
provided (Ollama, vLLM and most servers include it on the final SSE chunk);
otherwise we fall back to counting streamed content chunks as a rough proxy and
flag the count as estimated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Endpoint


@dataclass
class RequestResult:
    """Timing and output for a single inference request."""

    ok: bool
    endpoint: str
    model: str
    hardware: str
    task: str
    category: str = ""

    # Timing (seconds)
    ttft_s: float | None = None            # time to first generated token (reasoning or content)
    ttft_content_s: float | None = None    # time to first *answer* token (None if only reasoning)
    total_s: float | None = None           # request start -> stream end
    gen_s: float | None = None             # first generated token -> last token

    # Tokens
    prompt_tokens: int | None = None
    completion_tokens: int | None = None   # full generation incl. reasoning (from usage)
    tokens_estimated: bool = False         # True if completion_tokens is a proxy
    reasoning: bool = False                # model emitted reasoning/thinking tokens
    content_empty: bool = False            # generated tokens but no visible answer

    # Derived
    tokens_per_s: float | None = None      # completion_tokens / gen_s
    tps_from_total: bool = False           # tok/s fell back to /total_s (tiny gen window)

    # Output / diagnostics
    prompt: str = ""                       # the user prompt (for quality review)
    text: str = ""                         # visible answer content
    reasoning_text: str = ""               # hidden thinking (reasoning models)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    turns: int = 1                         # >1 for multi-turn agent tasks

    # Phase 2 scoring (filled in by aibench.scoring.run.score_results).
    quality: float | None = None           # overall quality in [0, 1]
    check_scores: dict[str, float] = field(default_factory=dict)
    score_note: str = ""

    # Phase 3 LLM-judge (filled in by aibench.scoring.judge).
    judge_score: float | None = None       # graduated quality in [0, 1] (from 1-5)
    judge_note: str = ""

    # A generation window shorter than this (absolute) OR shorter than this
    # fraction of the total request is treated as a burst: the tokens arrived
    # all at once (typically a batching/scheduling artifact under concurrency),
    # so completion_tokens/gen_s is not a real per-token decode rate and we
    # fall back to the end-to-end rate completion_tokens/total_s.
    _GEN_FLOOR_S = 0.05
    _GEN_MIN_FRACTION = 0.2

    def finalize(self) -> "RequestResult":
        # Reset derived fields so finalize() is idempotent — re-running it on a
        # loaded result recomputes tok/s with the current logic.
        self.tokens_per_s = None
        self.tps_from_total = False
        if not self.completion_tokens:
            return self
        floor = self._GEN_FLOOR_S
        if self.total_s:
            floor = max(floor, self._GEN_MIN_FRACTION * self.total_s)
        if self.gen_s and self.gen_s >= floor:
            self.tokens_per_s = self.completion_tokens / self.gen_s
        elif self.total_s and self.total_s > 0:
            self.tokens_per_s = self.completion_tokens / self.total_s
            self.tps_from_total = True
        return self


async def run_request(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    task_name: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    timeout_s: float,
    category: str = "",
    tools: list[dict[str, Any]] | None = None,
) -> RequestResult:
    """Execute one streaming chat-completion request and capture timing."""

    result = RequestResult(
        ok=False,
        endpoint=endpoint.name,
        model=endpoint.model,
        hardware=endpoint.hardware,
        task=task_name,
        category=category,
    )

    payload: dict[str, Any] = {
        "model": endpoint.model,
        "messages": messages,
        "stream": True,
        # Ask servers that support it to include token usage on the final chunk.
        "stream_options": {"include_usage": True},
    }
    payload.update(params)
    payload.update(endpoint.params)
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}
    key = endpoint.resolved_key
    if key and key != "not-needed":
        headers["Authorization"] = f"Bearer {key}"

    # Record the user prompt so reports are self-describing.
    result.prompt = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    chunk_count = 0            # content chunks
    reasoning_chunks = 0      # reasoning/thinking chunks
    start = time.perf_counter()
    first_token_at: float | None = None      # first token of any kind
    first_content_at: float | None = None    # first visible answer token
    saw_reasoning = False

    try:
        async with client.stream(
            "POST",
            endpoint.chat_url,
            json=payload,
            headers=headers,
            timeout=timeout_s,
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                result.error = f"HTTP {resp.status_code}: {body}"
                return result.finalize()

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Usage may arrive on its own final chunk (choices == []).
                usage = obj.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")

                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    # Reasoning models stream hidden thinking under a separate
                    # field (naming varies by server). These are generated
                    # tokens too, so they start the clock for TTFT/throughput.
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                    )
                    if reasoning:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        reasoning_parts.append(reasoning)
                        reasoning_chunks += 1
                        saw_reasoning = True
                    content = delta.get("content")
                    if content:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        if first_content_at is None:
                            first_content_at = now
                        text_parts.append(content)
                        chunk_count += 1
                    for tc in delta.get("tool_calls", []) or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function", {}) or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]

            end = time.perf_counter()
            result.ok = True
            result.total_s = end - start
            if first_token_at is not None:
                result.ttft_s = first_token_at - start
                result.gen_s = end - first_token_at
            if first_content_at is not None:
                result.ttft_content_s = first_content_at - start
            result.reasoning = saw_reasoning
            # A completed generation (or one that hit the length cap) that
            # produced no visible answer text spent its budget on reasoning
            # or tool calls only.
            result.content_empty = chunk_count == 0 and not tool_calls
            result.text = "".join(text_parts)
            result.reasoning_text = "".join(reasoning_parts)
            result.tool_calls = [
                {"id": v.get("id", ""), "name": v["name"], "arguments": v["arguments"]}
                for v in tool_calls.values()
            ]

            if result.completion_tokens is None and (chunk_count or reasoning_chunks):
                # Fall back to streamed-chunk count as a rough token proxy.
                result.completion_tokens = chunk_count + reasoning_chunks
                result.tokens_estimated = True

    except httpx.TimeoutException:
        result.error = f"timeout after {timeout_s}s"
    except httpx.HTTPError as e:
        result.error = f"{type(e).__name__}: {e}"

    return result.finalize()


async def probe_endpoint(endpoint: Endpoint, timeout_s: float = 15.0) -> tuple[bool, str]:
    """Lightweight reachability + model check for one endpoint.

    Sends a 1-token completion so we verify not just that the host is up but
    that the configured model actually answers. Returns (ok, human-readable
    detail) — used by the run preflight to detect dead endpoints before
    committing to a full benchmark.
    """
    body = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    key = endpoint.resolved_key
    if key and key != "not-needed":
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(endpoint.chat_url, json=body,
                                  headers=headers, timeout=timeout_s)
        if r.status_code == 200:
            return True, "OK"
        detail = (r.text or "").strip().replace("\n", " ")[:160]
        return False, f"HTTP {r.status_code}: {detail}"
    except httpx.HTTPError as e:
        return False, f"{type(e).__name__}: {e}"
