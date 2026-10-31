"""Append-only JSONL metrics: one line per round (or epoch), flushed immediately,
so a crashed or interrupted run still leaves a readable log."""
from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self.rows: list[dict] = []

    def log(self, **row) -> None:
        row.setdefault("time", round(time.time(), 3))
        self.rows.append(row)
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def summarize(history: list[dict], spec: dict, ok: bool, **extra) -> dict:
    """Summary numbers for one run. `history` rows have "round" and maybe "test_acc".

    final_acc_last5 (mean of the last five evaluations) is what the results
    table reports: under strong non-IID data the accuracy jumps around from
    round to round, so a single final value is mostly noise.
    """
    from fl.config import run_name
    evals = [h for h in history if h.get("test_acc") is not None]
    accs = [h["test_acc"] for h in evals]
    target = spec["target_acc"]
    return {
        "run_name": run_name(spec),
        "ok": ok,
        "rounds_completed": len(history),
        "final_acc": accs[-1] if accs else None,
        "final_acc_last5": sum(accs[-5:]) / len(accs[-5:]) if accs else None,
        "best_acc": max(accs) if accs else None,
        "rounds_to_target": next((h["round"] for h in evals if h["test_acc"] >= target), None),
        "target_acc": target,
        "total_time": round(sum(h.get("round_time", 0) for h in history), 1),
        **extra,
    }
