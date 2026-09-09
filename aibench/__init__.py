"""aibench — a local AI inference benchmarking tool.

Sends inference requests to OpenAI-compatible endpoints on your local network
and compares performance of models and hardware across task categories
(creative writing, code generation, summarization, agents/tool use).

Scoring is pluggable and phased:
  Phase 1 (current): performance metrics only (tokens/sec, TTFT, latency).
  Phase 2 (planned):  reference-based quality metrics.
  Phase 3 (optional): LLM-as-judge.
"""

__version__ = "0.1.0"
