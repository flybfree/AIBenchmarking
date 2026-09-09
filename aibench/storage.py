"""Persist and load benchmark runs as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .runner import RunOutput


def save_run(out: RunOutput, output_dir: str | Path) -> Path:
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (out.config.label or "run").replace(" ", "_")
    path = d / f"{label}_{stamp}.json"
    payload = {
        "config": out.config.to_dict(),
        "started_at": out.started_at,
        "finished_at": out.finished_at,
        "results": [asdict(r) for r in out.results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
