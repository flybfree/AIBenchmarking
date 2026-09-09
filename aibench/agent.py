"""Multi-turn agent loop.

Drives a tool-using conversation: the model is given tools, and whenever it
returns tool calls the harness executes them (via the task's `tool_impls`),
feeds the results back, and asks again — until the model produces a final
answer or `max_turns` is reached. This tests real agentic behaviour (choosing
tools, chaining tool outputs, synthesising an answer), not just whether a
single tool call is emitted.

The whole interaction is collapsed into one `RequestResult` so it flows through
the existing aggregation/scoring/report unchanged: `text` is the final answer,
`tool_calls` is every call made across turns, `turns` is the turn count, and
timing is summed across turns.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .client import RequestResult, run_request
from .config import Endpoint
from .tasks import Task


def _execute_tool(task: Task, name: str, arguments: str) -> str:
    impl = task.tool_impls.get(name)
    if impl is None:
        return f"Error: no such tool '{name}'."
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return f"Error: arguments were not valid JSON: {arguments!r}"
    try:
        return str(impl(args))
    except Exception as e:  # a bad tool impl shouldn't kill the run
        return f"Error running tool: {type(e).__name__}: {e}"


async def run_agent(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    task: Task,
    params: dict[str, Any],
    timeout_s: float,
) -> RequestResult:
    """Run the agent loop and return one aggregated RequestResult."""
    messages: list[dict[str, Any]] = task.messages()
    all_calls: list[dict[str, Any]] = []

    start = time.perf_counter()
    first_ttft: float | None = None
    comp_tokens = 0
    gen_s_sum = 0.0
    turns = 0
    final_text = ""
    last_err: str | None = None

    for _ in range(max(1, task.max_turns)):
        res = await run_request(
            client, endpoint, task.id, messages, params, timeout_s,
            category=task.category, tools=task.tools,
        )
        turns += 1
        if res.ttft_s is not None and first_ttft is None:
            first_ttft = res.ttft_s
        if res.completion_tokens:
            comp_tokens += res.completion_tokens
        if res.gen_s:
            gen_s_sum += res.gen_s
        if not res.ok:
            last_err = res.error
            break

        if res.tool_calls:
            # Echo the assistant turn (with its tool calls) then each result.
            assistant: dict[str, Any] = {"role": "assistant"}
            if res.text:
                assistant["content"] = res.text
            assistant["tool_calls"] = [
                {
                    "id": tc.get("id") or f"call_{len(all_calls) + i}",
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for i, tc in enumerate(res.tool_calls)
            ]
            messages.append(assistant)
            for ac, tc in zip(assistant["tool_calls"], res.tool_calls):
                result = _execute_tool(task, tc["name"], tc["arguments"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": ac["id"],
                    "content": result,
                })
            all_calls.extend(res.tool_calls)
            continue

        # No tool calls => the model's final answer.
        final_text = res.text
        break

    end = time.perf_counter()
    agg = RequestResult(
        ok=last_err is None,
        endpoint=endpoint.name,
        model=endpoint.model,
        hardware=endpoint.hardware,
        task=task.id,
        category=task.category,
        prompt=task.prompt,
        text=final_text,
        tool_calls=all_calls,
        turns=turns,
        error=last_err,
        ttft_s=first_ttft,
        total_s=end - start,
        gen_s=gen_s_sum or None,
        completion_tokens=comp_tokens or None,
        finish_reason="stop" if last_err is None else None,
    )
    return agg.finalize()
