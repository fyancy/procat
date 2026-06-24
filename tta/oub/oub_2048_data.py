"""OUB 2048 non-overlap trial-wise data pipeline.

Protocol:
- Window length 2048, step 2048 (non-overlapping), ~976 windows per 10s trial.
- Source train: trials 1+2, 5 classes; full = all windows, light = equispaced subsample.
- Source val: trial 3, 5 classes; full = all windows, light = equispaced subsample.
- Target test: ordered by domain -> class -> trial -> window; light modes subsample
  equispaced window indices from each trial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.oub import load_vibration_channel
from datasets.paths_oub import DEFAULT_CACHE_DIR, DEFAULT_RAW_DIR, get_oub_mat_path

INPUT_LENGTH = 2048
NUM_DOMAINS = 4
NUM_CLASSES = 5
NUM_TRIALS = 3
TRIAL_IDS: Tuple[int, ...] = (1, 2, 3)
TRAIN_TRIAL_IDS: Tuple[int, ...] = (1, 2)
VAL_TRIAL_ID = 3
WINDOWS_PER_TRIAL = 976

SUBSET_CHOICES: Tuple[str, ...] = ("light_trial1", "light", "full")
CACHE_FILENAME = "oub_len2048_nonoverlap_trialwise.npy"


def slice_nonoverlap(signal: np.ndarray, window: int = INPUT_LENGTH) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract all non-overlapping windows from a 1D signal."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size < window:
        raise ValueError(f"Signal too short: {signal.size} < {window}")
    step = window
    n_windows = (signal.size - window) // step + 1
    starts = np.arange(n_windows, dtype=np.int64) * step
    windows = np.stack([signal[start : start + window] for start in starts], axis=0)
    meta = {
        "signal_length": int(signal.size),
        "num_windows": int(n_windows),
        "window_length": int(window),
        "step": int(step),
    }
    return windows.astype(np.float32), meta


def equispaced_window_indices(n_total: int, n_pick: int) -> np.ndarray:
    """Pick ``n_pick`` equispaced indices from ``0 .. n_total-1``."""
    n_total = int(n_total)
    n_pick = int(n_pick)
    if n_pick <= 0:
        return np.empty((0,), dtype=np.int64)
    if n_pick >= n_total:
        return np.arange(n_total, dtype=np.int64)
    if n_pick == 1:
        return np.asarray([n_total // 2], dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, n_total - 1, n_pick)).astype(np.int64))


def _trial_to_index(trial: int) -> int:
    if trial not in TRIAL_IDS:
        raise ValueError(f"trial must be in {TRIAL_IDS}, got {trial}")
    return trial - 1


def _trials_for_subset(subset: str) -> Tuple[int, ...]:
    if subset == "light_trial1":
        return (1,)
    if subset in {"light", "full"}:
        return TRIAL_IDS
    raise ValueError(f"Unknown subset: {subset}. Expected one of {SUBSET_CHOICES}")


def _window_indices_for_subset(
    n_total: int,
    subset: str,
    light_samples: int,
) -> np.ndarray:
    if subset in {"light_trial1", "light"}:
        return equispaced_window_indices(n_total, light_samples)
    if subset == "full":
        return np.arange(n_total, dtype=np.int64)
    raise ValueError(f"Unknown subset: {subset}")


def _train_val_window_indices(subset: str, light_samples: int) -> np.ndarray:
    """Train/val windows: full for ``full`` subset, light subsample otherwise."""
    if subset == "full":
        return np.arange(WINDOWS_PER_TRIAL, dtype=np.int64)
    return equispaced_window_indices(WINDOWS_PER_TRIAL, light_samples)


def build_oub_2048_cache(
    raw_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Build or load trial-wise non-overlap cache.

    Returns:
        x: [domain, class, trial, window, 1, INPUT_LENGTH]
        y: [domain, class, trial, window]
        meta: dict with slice metadata
    """
    raw_path = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    cache_path_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = cache_path_root / CACHE_FILENAME

    if cache_path.is_file() and not force_rebuild:
        payload = np.load(cache_path, allow_pickle=True).item()
        return (
            payload["x"].astype(np.float32),
            payload["y"].astype(np.int32),
            dict(payload.get("meta", {})),
        )

    if not raw_path.exists():
        raise FileNotFoundError(f"OUB raw dir not found: {raw_path}")

    x = np.zeros(
        (NUM_DOMAINS, NUM_CLASSES, NUM_TRIALS, WINDOWS_PER_TRIAL, 1, INPUT_LENGTH),
        dtype=np.float32,
    )
    y = np.zeros((NUM_DOMAINS, NUM_CLASSES, NUM_TRIALS, WINDOWS_PER_TRIAL), dtype=np.int32)
    slice_meta: List[Dict[str, object]] = []

    for domain in range(NUM_DOMAINS):
        for label in range(NUM_CLASSES):
            for trial in TRIAL_IDS:
                trial_idx = _trial_to_index(trial)
                mat_path = get_oub_mat_path(raw_path, label, domain, trial)
                if not mat_path.exists():
                    raise FileNotFoundError(f"Missing OUB file: {mat_path}")
                signal = load_vibration_channel(mat_path)
                windows, win_meta = slice_nonoverlap(signal, window=INPUT_LENGTH)
                n_windows = int(windows.shape[0])
                if n_windows < WINDOWS_PER_TRIAL:
                    raise ValueError(
                        f"Expected at least {WINDOWS_PER_TRIAL} windows for "
                        f"domain={domain}, label={label}, trial={trial}, got {n_windows}"
                    )
                x[domain, label, trial_idx, :n_windows, 0, :] = windows[:WINDOWS_PER_TRIAL]
                y[domain, label, trial_idx, :n_windows] = label
                slice_meta.append(
                    {
                        "domain": domain,
                        "label": label,
                        "trial": trial,
                        **win_meta,
                    }
                )

    meta = {
        "input_length": INPUT_LENGTH,
        "overlap": 0.0,
        "windows_per_trial": WINDOWS_PER_TRIAL,
        "train_trials": list(TRAIN_TRIAL_IDS),
        "val_trial": VAL_TRIAL_ID,
        "policy": "non-overlap 2048-point windows, trial-wise storage",
        "slices": slice_meta,
    }
    cache_path_root.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, {"x": x, "y": y, "meta": meta})
    print(f"[DATA] saved {cache_path}, x={x.shape}")
    return x, y, meta


def _collect_domain_trials(
    x: np.ndarray,
    y: np.ndarray,
    domain: int,
    trial_ids: Sequence[int],
    window_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather windows for one domain across classes and trials."""
    x_out, y_out, _, _ = _collect_domain_trials_with_meta(x, y, domain, trial_ids, window_indices)
    return x_out, y_out


def _collect_domain_trials_with_meta(
    x: np.ndarray,
    y: np.ndarray,
    domain: int,
    trial_ids: Sequence[int],
    window_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather windows plus per-sample domain/trial ids."""
    parts_x: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    parts_d: List[np.ndarray] = []
    parts_t: List[np.ndarray] = []
    for label in range(NUM_CLASSES):
        for trial in trial_ids:
            trial_idx = _trial_to_index(trial)
            wx = x[domain, label, trial_idx, window_indices, 0, :]
            wy = y[domain, label, trial_idx, window_indices]
            n = int(wx.shape[0])
            parts_x.append(wx)
            parts_y.append(wy)
            parts_d.append(np.full(n, int(domain), dtype=np.int64))
            parts_t.append(np.full(n, int(trial), dtype=np.int64))
    x_out = np.concatenate(parts_x, axis=0).reshape(-1, 1, INPUT_LENGTH).astype(np.float32)
    y_out = np.concatenate(parts_y, axis=0).reshape(-1).astype(np.int32)
    d_out = np.concatenate(parts_d, axis=0).astype(np.int64)
    t_out = np.concatenate(parts_t, axis=0).astype(np.int64)
    return x_out, y_out, d_out, t_out


def get_oub_data_trialwise(
    source_domain: int,
    target_domains: Sequence[int],
    train_trials: Sequence[int] = (1, 2),
    test_trials: Sequence[int] = (1,),
    sample_len: int = INPUT_LENGTH,
    test_samples_per_trial: Optional[int] = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trial-wise OUB split API used by supps/oub eval pipeline."""
    if int(sample_len) != INPUT_LENGTH:
        raise ValueError(f"sample_len must be {INPUT_LENGTH}, got {sample_len}")

    x, y, _ = build_oub_2048_cache()
    if test_samples_per_trial is None or int(test_samples_per_trial) <= 0:
        train_window_idx = np.arange(WINDOWS_PER_TRIAL, dtype=np.int64)
        test_window_idx = np.arange(WINDOWS_PER_TRIAL, dtype=np.int64)
    else:
        train_window_idx = equispaced_window_indices(WINDOWS_PER_TRIAL, int(test_samples_per_trial))
        test_window_idx = equispaced_window_indices(WINDOWS_PER_TRIAL, int(test_samples_per_trial))

    x_train, y_train, d_train, t_train = _collect_domain_trials_with_meta(
        x, y, int(source_domain), train_trials, train_window_idx
    )

    test_x_parts: List[np.ndarray] = []
    test_y_parts: List[np.ndarray] = []
    test_d_parts: List[np.ndarray] = []
    test_t_parts: List[np.ndarray] = []
    for domain in target_domains:
        tx, ty, td, tt = _collect_domain_trials_with_meta(
            x, y, int(domain), test_trials, test_window_idx
        )
        test_x_parts.append(tx)
        test_y_parts.append(ty)
        test_d_parts.append(td)
        test_t_parts.append(tt)

    if test_x_parts:
        x_test = np.concatenate(test_x_parts, axis=0).astype(np.float32)
        y_test = np.concatenate(test_y_parts, axis=0).astype(np.int32)
        d_test = np.concatenate(test_d_parts, axis=0).astype(np.int64)
        t_test = np.concatenate(test_t_parts, axis=0).astype(np.int64)
    else:
        x_test = np.empty((0, 1, INPUT_LENGTH), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.int32)
        d_test = np.empty((0,), dtype=np.int64)
        t_test = np.empty((0,), dtype=np.int64)

    return x_train, y_train, x_test, y_test, d_test, t_test, d_train, t_train


def get_oub_2048_splits(
    source_domain: int,
    target_domains: Sequence[int],
    subset: str = "light_trial1",
    light_samples: int = 100,
    raw_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    force_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Build train/val/test arrays for one OUB cross-domain task."""
    if subset not in SUBSET_CHOICES:
        raise ValueError(f"Unknown subset: {subset}. Expected one of {SUBSET_CHOICES}")

    x, y, cache_meta = build_oub_2048_cache(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        force_rebuild=force_cache,
    )

    train_window_idx = _train_val_window_indices(subset, light_samples)
    val_window_idx = _train_val_window_indices(subset, light_samples)
    test_window_idx = _window_indices_for_subset(WINDOWS_PER_TRIAL, subset, light_samples)
    test_trials = _trials_for_subset(subset)

    x_train, y_train = _collect_domain_trials(
        x, y, int(source_domain), TRAIN_TRIAL_IDS, train_window_idx
    )
    x_val, y_val = _collect_domain_trials(
        x, y, int(source_domain), (VAL_TRIAL_ID,), val_window_idx
    )

    test_x_parts: List[np.ndarray] = []
    test_y_parts: List[np.ndarray] = []
    test_domain_id_parts: List[np.ndarray] = []
    for domain in target_domains:
        tx, ty = _collect_domain_trials(x, y, int(domain), test_trials, test_window_idx)
        test_x_parts.append(tx)
        test_y_parts.append(ty)
        test_domain_id_parts.append(np.full(ty.shape[0], int(domain), dtype=np.int64))

    x_test = np.concatenate(test_x_parts, axis=0).astype(np.float32)
    y_test = np.concatenate(test_y_parts, axis=0).astype(np.int32)
    test_domain_ids = np.concatenate(test_domain_id_parts, axis=0).astype(np.int64)

    split_meta = {
        **cache_meta,
        "subset": subset,
        "light_samples": int(light_samples),
        "source_domain": int(source_domain),
        "target_domains": [int(d) for d in target_domains],
        "train_trials": list(TRAIN_TRIAL_IDS),
        "val_trial": VAL_TRIAL_ID,
        "test_trials": list(test_trials),
        "train_window_indices": train_window_idx.tolist(),
        "val_window_indices": val_window_idx.tolist(),
        "test_window_indices": test_window_idx.tolist(),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "test_domain_ids": test_domain_ids.tolist(),
    }
    return x_train, y_train, x_val, y_val, x_test, y_test, split_meta
