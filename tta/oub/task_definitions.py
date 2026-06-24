"""OUB cross-domain task definitions (T1-T8)."""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

DOMAIN_NAMES: Tuple[str, ...] = ("B1_inc", "B2_dec", "B3_inc_dec", "B4_dec_inc")

OUB_TASKS: Dict[str, Dict[str, object]] = {
    "T1": {"source": 0, "targets": [1, 2, 3], "gamma": 100.0},
    "T2": {"source": 1, "targets": [0, 2, 3], "gamma": 100.0},
    "T3": {"source": 2, "targets": [0, 1, 3], "gamma": 10.0},
    "T4": {"source": 3, "targets": [0, 1, 2], "gamma": 10.0},
    "T5": {"source": 0, "targets": [1, 2, 3], "gamma": 1.0},
    "T6": {"source": 1, "targets": [0, 2, 3], "gamma": 1.0},
    "T7": {"source": 2, "targets": [0, 1, 3], "gamma": 0.1},
    "T8": {"source": 3, "targets": [0, 1, 2], "gamma": 0.1},
}

DEFAULT_METHODS: Tuple[str, ...] = (
    "baseline",
    "bn_adapt",
    "rotta",
    "tribe",)


def parse_tasks_arg(value: str) -> List[str]:
    text = (value or "").strip().lower()
    if not text or text == "all":
        return list(OUB_TASKS.keys())
    out: List[str] = []
    for part in value.replace(" ", ",").split(","):
        part = part.strip().upper()
        if not part:
            continue
        if part not in OUB_TASKS:
            raise ValueError(f"Unknown OUB task: {part}")
        out.append(part)
    return out


def checkpoint_for_domain(model: str, source_domain: int) -> str:
    """[DEPRECATED] Old clean_noaug checkpoint path; use run_oub_2048_full.checkpoint_for_domain."""
    warnings.warn(
        "task_definitions.checkpoint_for_domain() is DEPRECATED (clean_noaug checkpoints). "
        "Use tta.oub.run_oub_2048_full.checkpoint_for_domain() for len2048 trial-split models.",
        DeprecationWarning,
        stacklevel=2,
    )
    return f"checkpoints/{model}_oub_B{source_domain + 1}_clean_noaug.pth"
