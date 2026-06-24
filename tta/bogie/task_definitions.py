"""Bogie multi-speed train/test task definitions (T1-T6)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from tta.bogie.bogie_data import SPEEDS

SPEED_NAMES: Tuple[str, ...] = tuple(f"S{i}_rpm{rpm}" for i, rpm in enumerate(SPEEDS))


def _speed_label(idxs: Sequence[int]) -> str:
    return "+".join(SPEED_NAMES[i] for i in idxs)


BOGIE_TASKS: Dict[str, Dict[str, object]] = {
    "T1": {
        "sources": [0, 1],
        "source_rpms": [1000, 1500],
        "targets": [0, 1, 2],
        "target_rpms": [1000, 1500, 2000],
        "gamma": 100.0,
    },
    "T2": {
        "sources": [0, 2],
        "source_rpms": [1000, 2000],
        "targets": [0, 2, 1],
        "target_rpms": [1000, 2000, 1500],
        "gamma": 100.0,
    },
    "T3": {
        "sources": [1, 2],
        "source_rpms": [1500, 2000],
        "targets": [1, 2, 0],
        "target_rpms": [1500, 2000, 1000],
        "gamma": 100.0,
    },
    "T4": {
        "sources": [0, 1],
        "source_rpms": [1000, 1500],
        "targets": [0, 1, 2],
        "target_rpms": [1000, 1500, 2000],
        "gamma": 0.1,
    },
    "T5": {
        "sources": [0, 2],
        "source_rpms": [1000, 2000],
        "targets": [0, 2, 1],
        "target_rpms": [1000, 2000, 1500],
        "gamma": 0.1,
    },
    "T6": {
        "sources": [1, 2],
        "source_rpms": [1500, 2000],
        "targets": [1, 2, 0],
        "target_rpms": [1500, 2000, 1000],
        "gamma": 0.1,
    },
}


def parse_tasks_arg(value: str) -> List[str]:
    text = (value or "").strip().lower()
    if not text or text == "all":
        return list(BOGIE_TASKS.keys())
    out: List[str] = []
    for part in value.replace(" ", ",").split(","):
        part = part.strip().upper()
        if not part:
            continue
        if part not in BOGIE_TASKS:
            raise ValueError(f"Unknown Bogie task: {part}")
        out.append(part)
    return out


def sources_label(source_idxs: Sequence[int]) -> str:
    return _speed_label(source_idxs)
