"""SQ three-scenario experiment definitions (domain / noise_dyn / gli)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

SPEED_HZ: Dict[int, int] = {0: 9, 1: 19, 2: 29, 3: 39}

DEFAULT_TRAIN_SPEEDS: Tuple[int, ...] = (0, 1)
DEFAULT_DOMAIN_TEST_SPEEDS: Tuple[int, ...] = (2, 3)
DEFAULT_CORRUPTION_SPEEDS: Tuple[int, ...] = (0, 1)
DEFAULT_GLI_GAMMA: float = 0.1

SQ_NUM_CLASSES: int = 7
SQ_INPUT_LENGTH: int = 2048

SQ_SCENARIOS: Dict[str, Dict[str, object]] = {
    "domain": {
        "id": "domain",
        "result_key": "domain",
        "label_zh": "跨域",
        "train_speeds": list(DEFAULT_TRAIN_SPEEDS),
        "test_speeds": list(DEFAULT_DOMAIN_TEST_SPEEDS),
        "corruption_type": None,
        "gli_gamma": None,
        "extra_hint": "",
        "description": (
            "Cross-speed domain shift: source train on S0+S1 (9/19 Hz), "
            "online test stream on S2+S3 (29/39 Hz)."
        ),
    },
    "noise_dyn": {
        "id": "noise_dyn",
        "result_key": "noise_dyn",
        "label_zh": "动态噪声",
        "train_speeds": list(DEFAULT_TRAIN_SPEEDS),
        "test_speeds": list(DEFAULT_CORRUPTION_SPEEDS),
        "corruption_type": "noise_dyn",
        "gli_gamma": None,
        "extra_hint": "snr=10->0, levels=11",
        "description": (
            "Dynamic noise corruption on S0+S1: SNR decreases from 10 dB to 0 dB "
            "across 11 levels during the online test stream."
        ),
    },
    "gli": {
        "id": "gli",
        "result_key": "gli_gamma0.1",
        "label_zh": "Dirichlet 标签偏移",
        "train_speeds": list(DEFAULT_TRAIN_SPEEDS),
        "test_speeds": list(DEFAULT_DOMAIN_TEST_SPEEDS),
        "corruption_type": None,
        "gli_gamma": DEFAULT_GLI_GAMMA,
        "extra_hint": "gamma=0.1",
        "description": (
            "Cross-speed domain shift (S0+S1 train, S2+S3 test) with per-domain "
            "Dirichlet label imbalance (gamma=0.1)."
        ),
    },
}

SCENARIO_ORDER: Tuple[str, ...] = ("domain", "noise_dyn", "gli_gamma0.1")


def sq_speed_label(speed_idx: int) -> str:
    hz = SPEED_HZ.get(int(speed_idx), "?")
    return f"S{speed_idx}({hz}Hz)"


def speed_label(idxs: Sequence[int]) -> str:
    return "+".join(sq_speed_label(int(i)) for i in idxs)


def scenario_by_result_key(result_key: str) -> Optional[Dict[str, object]]:
    for spec in SQ_SCENARIOS.values():
        if spec["result_key"] == result_key:
            return spec
    return None


def format_scenario_definitions_table() -> List[str]:
    lines = [
        "| Scenario ID | 中文名 | Train domain | Test domain | Corruption / GLI | CSV extra |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("domain", "noise_dyn", "gli"):
        spec = SQ_SCENARIOS[key]
        train = speed_label(spec["train_speeds"])  # type: ignore[arg-type]
        test = speed_label(spec["test_speeds"])  # type: ignore[arg-type]
        corr = spec["corruption_type"] or "-"
        if spec.get("gli_gamma") is not None:
            corr = f"Dirichlet γ={spec['gli_gamma']}"
        extra = spec.get("extra_hint") or "-"
        lines.append(
            f"| `{spec['result_key']}` | {spec['label_zh']} | {train} | {test} | {corr} | {extra} |"
        )
    return lines


def format_proto_config_block(cfg: Mapping[str, object]) -> List[str]:
    return []


def format_tune_setup_header(
    *,
    model: str,
    method_config: Path,
    objective: str,
    eval_logits_timing: str = "pre_adapt",
    min_delta_pp: Optional[float] = None,
    extra_notes: Optional[List[str]] = None,
) -> str:
    lines = [
        f"# SQ Proto tuning ({model})",
        "",
        "## Experiment Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| dataset | SQ |",
        f"| model | {model} |",
        f"| scenarios | domain (跨域), noise_dyn (动态噪声), gli (Dirichlet γ=0.1) |",
        f"| method_config | `{method_config}` |",
        f"| eval_logits_timing | {eval_logits_timing} |",
        f"| tuning objective | {objective} |",
    ]
    if min_delta_pp is not None:
        lines.append(f"| min_delta_pp | {min_delta_pp} |")
    lines.append("")
    lines.extend(format_scenario_definitions_table())
    lines.append("")
    if extra_notes:
        lines.append("### Notes")
        lines.append("")
        for note in extra_notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def load_method_config_json(config_path: Path) -> Dict[str, object]:
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))
