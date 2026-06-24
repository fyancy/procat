"""Bogie cross-speed data splits for TTA (channel 0, 2048 windows)."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from datasets.paths_bogie import NPY_TEMPLATE, NUMPY_DATA_DIR, SPEEDS
from tta.dirichlet_imbalance import build_dirichlet_domain_sequence

INPUT_LENGTH = 2048
NUM_CLASSES = 10
NUM_SPEEDS = 3
SAMPLES_PER_CLASS = 180


def rpm_for_speed_idx(speed_idx: int) -> int:
    return int(SPEEDS[int(speed_idx)])


def load_rpm_npy(speed: int, channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load one rpm npy; return x (N, 1, 2048), y (N,)."""
    path = os.path.join(NUMPY_DATA_DIR, NPY_TEMPLATE.format(speed=int(speed)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing Bogie npy: {path}")
    payload = np.load(path, allow_pickle=True).item()
    x = np.asarray(payload["x"], dtype=np.float32)
    y = np.asarray(payload["y"], dtype=np.int32).reshape(-1)
    if x.ndim != 3:
        raise ValueError(f"Expected x (N, L, C), got {x.shape}")
    if not (0 <= channel < x.shape[2]):
        raise ValueError(f"channel {channel} out of range for C={x.shape[2]}")
    x_ch = x[:, :, channel][:, np.newaxis, :]
    return x_ch.astype(np.float32), y


def stratified_train_val_indices(
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    rng = np.random.RandomState(int(seed))
    train_parts: List[int] = []
    val_parts: List[int] = []
    for cls in range(int(labels.max()) + 1):
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            continue
        perm = idx.copy()
        rng.shuffle(perm)
        n_train = int(round(perm.size * float(train_ratio)))
        n_train = min(max(n_train, 1), perm.size - 1) if perm.size > 1 else perm.size
        train_parts.extend(perm[:n_train].tolist())
        val_parts.extend(perm[n_train:].tolist())
    return np.asarray(train_parts, dtype=np.int64), np.asarray(val_parts, dtype=np.int64)


def load_multi_rpm_npy(
    source_speed_idxs: Sequence[int],
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合并多转速 npy，返回 x, y, speed_domain_ids。"""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ds: List[np.ndarray] = []
    for s_idx in source_speed_idxs:
        rpm = rpm_for_speed_idx(int(s_idx))
        x_part, y_part = load_rpm_npy(rpm, channel=channel)
        xs.append(x_part)
        ys.append(y_part)
        ds.append(np.full(y_part.shape[0], int(s_idx), dtype=np.int64))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int32)
    domain_ids = np.concatenate(ds, axis=0).astype(np.int64)
    return x, y, domain_ids


def _source_val_local_indices(
    domain_ids: np.ndarray,
    val_idx: np.ndarray,
    source_speed_idxs: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Per-source-speed val indices local to each rpm npy block (same split as train/val)."""
    offsets: Dict[int, int] = {}
    offset = 0
    for s_idx in source_speed_idxs:
        s_idx = int(s_idx)
        offsets[s_idx] = offset
        offset += int(np.sum(domain_ids == s_idx))
    out: Dict[int, np.ndarray] = {}
    for s_idx in source_speed_idxs:
        s_idx = int(s_idx)
        global_val = val_idx[domain_ids[val_idx] == s_idx]
        out[s_idx] = (global_val - offsets[s_idx]).astype(np.int64)
    return out


def get_bogie_source_splits(
    source_speed_idxs: Sequence[int],
    train_ratio: float = 0.8,
    seed: int = 0,
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    source_speed_idxs = [int(s) for s in source_speed_idxs]
    x_all, y_all, domain_ids = load_multi_rpm_npy(source_speed_idxs, channel=channel)
    train_idx, val_idx = stratified_train_val_indices(y_all, train_ratio=train_ratio, seed=seed)
    source_val_local = _source_val_local_indices(domain_ids, val_idx, source_speed_idxs)
    meta = {
        "source_speed_idxs": source_speed_idxs,
        "source_rpms": [rpm_for_speed_idx(s) for s in source_speed_idxs],
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "source_val_local_idx": source_val_local,
        "channel": int(channel),
    }
    return x_all[train_idx], y_all[train_idx], x_all[val_idx], y_all[val_idx], meta


def get_bogie_splits(
    source_speed_idxs: Sequence[int],
    target_speed_idxs: Sequence[int],
    train_ratio: float = 0.8,
    gamma: float = 100.0,
    seed: int = 0,
    channel: int = 0,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    source_speed_idxs = [int(s) for s in source_speed_idxs]
    x_train, y_train, x_val, y_val, src_meta = get_bogie_source_splits(
        source_speed_idxs, train_ratio=train_ratio, seed=seed, channel=channel
    )

    target_speed_idxs = [int(t) for t in target_speed_idxs]
    source_set = set(source_speed_idxs)
    source_val_local: Dict[int, np.ndarray] = src_meta["source_val_local_idx"]
    test_x_parts: List[np.ndarray] = []
    test_y_parts: List[np.ndarray] = []
    test_d_parts: List[np.ndarray] = []
    test_samples_by_speed: Dict[int, int] = {}

    for t_idx in target_speed_idxs:
        rpm = rpm_for_speed_idx(t_idx)
        x_t, y_t = load_rpm_npy(rpm, channel=channel)
        if t_idx in source_set:
            local_idx = source_val_local[t_idx]
            x_part = x_t[local_idx]
            y_part = y_t[local_idx]
        else:
            x_part = x_t
            y_part = y_t
        test_x_parts.append(x_part)
        test_y_parts.append(y_part)
        test_d_parts.append(np.full(y_part.shape[0], t_idx, dtype=np.int64))
        test_samples_by_speed[t_idx] = int(y_part.shape[0])

    x_test = np.concatenate(test_x_parts, axis=0).astype(np.float32)
    y_test = np.concatenate(test_y_parts, axis=0).astype(np.int32)
    test_domain_ids = np.concatenate(test_d_parts, axis=0).astype(np.int64)

    order, stream_stats = build_dirichlet_domain_sequence(
        labels=y_test,
        domain_ids=test_domain_ids,
        domain_order=target_speed_idxs,
        gamma=float(gamma),
        seed=int(seed),
        head_window=int(batch_size),
    )

    meta = {
        **src_meta,
        "target_speed_idxs": target_speed_idxs,
        "target_rpms": [rpm_for_speed_idx(t) for t in target_speed_idxs],
        "gamma": float(gamma),
        "test_samples": int(x_test.shape[0]),
        "test_samples_by_speed": test_samples_by_speed,
        "test_domain_ids": test_domain_ids,
        "dirichlet_order": order,
        "stream_stats": stream_stats,
    }
    return x_train, y_train, x_val, y_val, x_test, y_test, meta
