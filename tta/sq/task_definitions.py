"""SQ eight-task definitions: T1–T6 cross-domain speed pairs, T7 noise_dyn, T8 gli."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tta.sq.scenario_definitions import DEFAULT_GLI_GAMMA, speed_label, sq_speed_label

ROOT_DIR = Path(__file__).resolve().parents[2]
ALL_SPEEDS: Tuple[int, ...] = (0, 1, 2, 3)

LEGACY_SCENARIO_TO_TASK = {
    "domain": "T1",
    "noise_dyn": "T7",
    "gli": "T8",
    "gli_gamma0.1": "T8",
}

TASK_TO_LEGACY_SCENARIO = {
    "T1": "domain",
    "T7": "noise_dyn",
    "T8": "gli",
}


def _build_cross_domain_tasks() -> Dict[str, Dict[str, object]]:
    tasks: Dict[str, Dict[str, object]] = {}
    for idx, sources in enumerate(combinations(ALL_SPEEDS, 2), start=1):
        src = sorted(sources)
        tgt = sorted(s for s in ALL_SPEEDS if s not in src)
        tasks[f"T{idx}"] = {
            "task_id": f"T{idx}",
            "task_type": "cross_domain",
            "label_zh": "跨域",
            "sources": src,
            "targets": tgt,
            "corruption_type": None,
            "gli_gamma": None,
            "extra_hint": "",
            "description": (
                f"Cross-speed domain shift: train on {speed_label(src)}, "
                f"online test on {speed_label(tgt)}."
            ),
        }
    return tasks


SQ_TASKS: Dict[str, Dict[str, object]] = {
    **_build_cross_domain_tasks(),
    "T7": {
        "task_id": "T7",
        "task_type": "noise_dyn",
        "label_zh": "动态噪声",
        "sources": [0, 1],
        "targets": [0, 1],
        "corruption_type": "noise_dyn",
        "gli_gamma": None,
        "extra_hint": "snr=10->0, levels=11",
        "description": (
            "Dynamic noise on S0+S1: SNR decreases from 10 dB to 0 dB across 11 levels."
        ),
    },
    "T8": {
        "task_id": "T8",
        "task_type": "gli",
        "label_zh": "Dirichlet 标签偏移",
        "sources": [0, 1],
        "targets": [2, 3],
        "corruption_type": None,
        "gli_gamma": DEFAULT_GLI_GAMMA,
        "extra_hint": f"gamma={DEFAULT_GLI_GAMMA}",
        "description": (
            f"Cross-speed (S0+S1 train, S2+S3 test) with Dirichlet label imbalance "
            f"(gamma={DEFAULT_GLI_GAMMA})."
        ),
    },
}

SQ_TASK_ORDER: Tuple[str, ...] = tuple(f"T{i}" for i in range(1, 9))
CROSS_DOMAIN_TASKS: Tuple[str, ...] = tuple(f"T{i}" for i in range(1, 7))


def parse_tasks_arg(value: str) -> List[str]:
    text = (value or "").strip().lower()
    if not text or text == "all":
        return list(SQ_TASK_ORDER)
    out: List[str] = []
    for part in value.replace(" ", ",").split(","):
        part = part.strip().upper()
        if not part:
            continue
        if part not in SQ_TASKS:
            raise ValueError(f"Unknown SQ task: {part}. Supported: {', '.join(SQ_TASK_ORDER)}")
        out.append(part)
    return out


def sources_tag(source_speed_idxs: Sequence[int]) -> str:
    return "_".join(f"S{int(s)}" for s in sorted(source_speed_idxs))


def checkpoint_for_sources(model: str, source_speed_idxs: Sequence[int]) -> str:
    tag = sources_tag(source_speed_idxs)
    per_task = ROOT_DIR / "checkpoints" / f"{model}_sq_train_{tag}.pth"
    if tag == "S0_S1":
        legacy = ROOT_DIR / "checkpoints" / f"{model}_sq_clean_noaug.pth"
        if legacy.is_file():
            return str(legacy)
    return str(per_task)


def checkpoint_for_task(model: str, task_id: str) -> str:
    spec = SQ_TASKS[task_id]
    return checkpoint_for_sources(model, spec["sources"])  # type: ignore[arg-type]


def task_result_key(task_id: str, extra: str = "") -> str:
    if task_id == "T8" and extra.startswith("gamma="):
        return "T8"
    return task_id


def proto_scenario_label(task_id: str, extra: str = "") -> str:
    legacy = TASK_TO_LEGACY_SCENARIO.get(task_id)
    if legacy == "gli":
        return "gli_gamma0.1"
    if legacy:
        return legacy
    return task_id


def format_task_definitions_table() -> List[str]:
    lines = [
        "| Task | 类型 | Train domain | Test domain | Corruption / GLI |",
        "|---|---|---|---|---|",
    ]
    for task_id in SQ_TASK_ORDER:
        spec = SQ_TASKS[task_id]
        train = speed_label(spec["sources"])  # type: ignore[arg-type]
        test = speed_label(spec["targets"])  # type: ignore[arg-type]
        corr = spec.get("corruption_type") or "-"
        if spec.get("gli_gamma") is not None:
            corr = f"Dirichlet γ={spec['gli_gamma']}"
        label = spec["label_zh"]
        lines.append(f"| `{task_id}` | {label} | {train} | {test} | {corr} |")
    return lines


def task_by_result_key(result_key: str) -> Dict[str, object] | None:
    if result_key in SQ_TASKS:
        return SQ_TASKS[result_key]
    mapped = LEGACY_SCENARIO_TO_TASK.get(result_key)
    if mapped:
        return SQ_TASKS[mapped]
    return None
