"""Configuration loading and data models.

A benchmark run is described by a YAML/JSON file listing the endpoints to test
and which task suites to run. Endpoints speak the OpenAI-compatible
`/v1/chat/completions` API (Ollama, LM Studio, vLLM, llama.cpp server, TGI, ...).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Endpoint:
    """A single inference endpoint to benchmark.

    `base_url` is the OpenAI-compatible root, e.g. "http://192.168.1.50:11434/v1".
    `hardware` is a free-form label (e.g. "RTX 4090", "M2 Max") used only for
    reporting/grouping so you can compare the same model across machines.
    """

    name: str
    base_url: str
    model: str
    api_key: str = "not-needed"
    hardware: str = ""
    # Per-endpoint overrides for generation params (merged over task defaults).
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def resolved_key(self) -> str:
        """The API key to send. If api_key is written as `env:VAR`, `${VAR}` or
        `$VAR`, the value is read from that environment variable at runtime so
        the secret never lives in the config file. Otherwise it's used literally
        ('not-needed' means send no Authorization header)."""
        v = (self.api_key or "").strip()
        m = re.fullmatch(r"env:(\w+)|\$\{(\w+)\}|\$(\w+)", v)
        if m:
            name = next(g for g in m.groups() if g)
            return os.environ.get(name, "")
        return v


@dataclass
class RunConfig:
    """Top-level configuration for a benchmark run."""

    endpoints: list[Endpoint]
    tasks: list[str]                       # task-suite names, or ["all"]
    repeats: int = 3                       # runs per (endpoint, task) for averaging
    warmup: int = 1                        # discarded warmup runs (model load, cache)
    timeout_s: float = 300.0               # per-request timeout
    concurrency: int = 1                   # concurrent requests per endpoint
    parallel_endpoints: bool = False       # run endpoints at once (separate machines only)
    max_tokens: int | None = None          # answer budget; falls back to task default
    reasoning_reserve: int = 8192          # extra tokens added for thinking room
    #   (heavy reasoners can spend 4k+ tokens thinking before the answer)
    output_dir: str = "results"
    label: str = ""                        # optional label for this run
    # Optional Phase 3 LLM-judge endpoint (OpenAI-compatible). When set, judged
    # (subjective) tasks get a graduated 1-5 rubric score. Point it at a strong
    # model; it need not be one of the endpoints under test.
    judge: Endpoint | None = None

    @staticmethod
    def load(path: str | Path) -> "RunConfig":
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        data: dict[str, Any]
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        return RunConfig.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RunConfig":
        eps = [Endpoint(**e) for e in data.get("endpoints", [])]
        if not eps:
            raise ValueError("Config must define at least one endpoint.")
        known = {f for f in RunConfig.__dataclass_fields__ if f != "endpoints"}
        kwargs = {k: v for k, v in data.items() if k in known}
        if isinstance(kwargs.get("judge"), dict):
            kwargs["judge"] = Endpoint(**kwargs["judge"])
        return RunConfig(endpoints=eps, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
