"""Configurable dataset paths for the TTA benchmark package."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

_PKG_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = Path(r"H:\Datasets")

_RELATIVE_DEFAULTS: Dict[str, str] = {
    "sq_npy_resampled": "SQdata/numpy_data_resampled/sq_no_noise_resampled_for_att.npy",
    "sq_npy_unresampled": "SQdata/numpy_data_unresampled/sq_data_raw_250302_ntu.npy",
    "sq_legacy_dir": "dataset_MDA/SQ",
    "sq_raw_home": "SQdata",
    "bogie_root": "转向架齿轮轴承",
    "oub_raw": "OUBdata/raw",
    "oub_cache": "OUBdata/numpy_resampled",
}


def package_root() -> Path:
    return _PKG_ROOT


def _load_yaml_paths() -> Dict[str, str]:
    yaml_path = _PKG_ROOT / "configs" / "paths.yaml"
    if not yaml_path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(yaml_path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def _parse_simple_yaml(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            out[key.strip()] = value
    return out


@lru_cache(maxsize=1)
def get_paths_config() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    env_root = os.environ.get("TTA_DATA_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
        for key, rel in _RELATIVE_DEFAULTS.items():
            merged[key] = str(root / rel)
    merged.update(_load_yaml_paths())
    return merged


def resolve_data_path(key: str, fallback: Optional[str] = None) -> Path:
    cfg = get_paths_config()
    if key in cfg:
        return Path(cfg[key])
    if fallback is not None:
        return Path(fallback)
    rel = _RELATIVE_DEFAULTS.get(key)
    if rel is not None:
        return _DEFAULT_DATA_ROOT / rel
    raise KeyError(f"Unknown data path key: {key}")


def sq_npy_path(*, resample: bool = True) -> Path:
    key = "sq_npy_resampled" if resample else "sq_npy_unresampled"
    return resolve_data_path(key)


def sq_legacy_dir() -> Path:
    return resolve_data_path("sq_legacy_dir", str(_DEFAULT_DATA_ROOT / "dataset_MDA" / "SQ"))


def sq_raw_home() -> Path:
    return resolve_data_path("sq_raw_home", str(_DEFAULT_DATA_ROOT / "SQdata"))


def bogie_root() -> Path:
    return resolve_data_path("bogie_root", str(_DEFAULT_DATA_ROOT / "转向架齿轮轴承"))


def oub_raw_dir() -> Path:
    return resolve_data_path("oub_raw", str(_DEFAULT_DATA_ROOT / "OUBdata" / "raw"))


def oub_cache_dir() -> Path:
    return resolve_data_path("oub_cache", str(_DEFAULT_DATA_ROOT / "OUBdata" / "numpy_resampled"))

