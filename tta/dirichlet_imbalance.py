"""Dirichlet-based label imbalance ordering for TTA streams.

Reference: TRIBE ``LabelDirichletDomainSequence`` (gamma = concentration).
Lower gamma -> more skewed class proportions within each stream block.

OUB cross-domain tasks: apply Dirichlet **inside each target domain block**,
then concatenate blocks in ``target_domain_order`` (TTA state continuous, label
shift re-drawn per domain with an independent seed).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset

try:
    from .common import DataConfig, LoaderConfig, create_sq_datasets
except ImportError:
    from common import DataConfig, LoaderConfig, create_sq_datasets


def _dirichlet_reorder_block(
    labels: np.ndarray,
    block_indices: np.ndarray,
    gamma: float,
    rng: np.random.RandomState,
    num_slots: Optional[int] = None,
) -> np.ndarray:
    """TRIBE-style Dirichlet reorder within one index block."""
    block_indices = np.asarray(block_indices, dtype=np.int64).reshape(-1)
    if block_indices.size == 0:
        return block_indices

    block_labels = labels[block_indices]
    num_classes = int(labels.max()) + 1
    slots = num_classes if num_slots is None else int(num_slots)

    local_pos = np.arange(block_indices.size, dtype=np.int64)
    class_pos = [local_pos[block_labels == y] for y in range(num_classes)]

    slot_groups: List[List[np.ndarray]] = [[] for _ in range(slots)]
    label_distribution = rng.dirichlet([gamma] * slots, num_classes)

    for c_pos, partition in zip(class_pos, label_distribution):
        if c_pos.size == 0:
            continue
        split_points = (np.cumsum(partition)[:-1] * c_pos.size).astype(int)
        for slot_id, ids in enumerate(np.split(c_pos, split_points)):
            slot_groups[slot_id].append(ids)

    ordered_local: List[int] = []
    for slot_id in range(slots):
        groups = slot_groups[slot_id]
        if not groups:
            continue
        for gi in rng.permutation(len(groups)):
            ordered_local.extend(groups[gi].tolist())

    return block_indices[np.asarray(ordered_local, dtype=np.int64)]


def build_dirichlet_indices_within_subset(
    labels: np.ndarray,
    subset_indices: np.ndarray,
    gamma: float,
    num_slots: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Dirichlet reorder restricted to ``subset_indices`` (global index array)."""
    rng = np.random.RandomState(int(seed))
    return _dirichlet_reorder_block(
        labels,
        np.asarray(subset_indices, dtype=np.int64),
        gamma,
        rng,
        num_slots=num_slots,
    )


def build_sequential_dirichlet_domain_indices(
    labels: np.ndarray,
    domain_ids: np.ndarray,
    target_domain_order: List[int],
    gamma: float,
    num_slots: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """Sequential multi-domain stream with per-domain Dirichlet label shift.

    Domains are processed in ``target_domain_order``; within each domain, samples
    are reordered by Dirichlet class proportions (concentration ``gamma``).
    Each domain uses an independent derived seed so draws do not depend on prior
    domains' RNG consumption.
    """
    labels = np.asarray(labels).reshape(-1)
    domain_ids = np.asarray(domain_ids).reshape(-1)
    final: List[int] = []
    boundaries: List[int] = []
    for domain_idx, domain in enumerate(target_domain_order):
        domain = int(domain)
        mask = domain_ids == domain
        subset = np.where(mask)[0]
        if len(subset) == 0:
            continue
        domain_seed = int(seed) + domain_idx * 997 + domain * 13
        ordered = build_dirichlet_indices_within_subset(
            labels=labels,
            subset_indices=subset,
            gamma=gamma,
            num_slots=num_slots,
            seed=domain_seed,
        )
        final.extend(ordered.tolist())
        boundaries.append(len(final))
    return np.asarray(final, dtype=np.int64), boundaries


def build_dirichlet_indices(
    labels: np.ndarray,
    gamma: float,
    num_slots: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Reorder sample indices with a single global Dirichlet block (SQ / GLI)."""
    labels = np.asarray(labels).reshape(-1)
    all_indices = np.arange(len(labels), dtype=np.int64)
    return build_dirichlet_indices_within_subset(
        labels=labels,
        subset_indices=all_indices,
        gamma=gamma,
        num_slots=num_slots,
        seed=seed,
    )


def build_dirichlet_domain_sequence(
    labels: np.ndarray,
    domain_ids: np.ndarray,
    domain_order: Sequence[int],
    gamma: float,
    num_slots: Optional[int] = None,
    seed: int = 0,
    head_window: int = 64,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Per-target-domain Dirichlet reorder with stream stats."""
    labels = np.asarray(labels).reshape(-1)
    domain_ids = np.asarray(domain_ids).reshape(-1)
    if labels.shape[0] != domain_ids.shape[0]:
        raise ValueError("labels and domain_ids must have the same length")

    order_list = [int(d) for d in domain_order]
    order, end_boundaries = build_sequential_dirichlet_domain_indices(
        labels=labels,
        domain_ids=domain_ids,
        target_domain_order=order_list,
        gamma=gamma,
        num_slots=num_slots,
        seed=seed,
    )

    num_classes = int(labels.max()) + 1
    start = 0
    boundary_stats: List[Dict[str, object]] = []
    for domain, end in zip(order_list, end_boundaries):
        domain_labels = labels[order[start:end]]
        head = domain_labels[: min(head_window, domain_labels.size)]
        head_counts = {int(c): int((head == c).sum()) for c in range(num_classes)}
        boundary_stats.append(
            {
                "domain": int(domain),
                "start": start,
                "end": int(end),
                "num_samples": int(end - start),
                "head_counts": head_counts,
            }
        )
        start = int(end)

    stats = summarize_stream_class_counts(labels, order, window=head_window)
    stats.update(
        {
            "gamma": float(gamma),
            "seed": int(seed),
            "domain_order": order_list,
            "boundaries": boundary_stats,
            "boundary_ends": [int(x) for x in end_boundaries],
            "dirichlet_mode": "per_domain",
        }
    )
    return order, stats


def infer_equal_domain_ids(domain_order: Sequence[int], total_samples: int) -> np.ndarray:
    """Build domain id array when each domain contributes equal sample counts."""
    domain_order = [int(d) for d in domain_order]
    n_domains = len(domain_order)
    if n_domains == 0 or total_samples % n_domains != 0:
        raise ValueError(
            f"Cannot infer equal domain ids: total={total_samples}, domains={n_domains}"
        )
    n_per = total_samples // n_domains
    parts = [np.full(n_per, d, dtype=np.int64) for d in domain_order]
    return np.concatenate(parts)


def summarize_stream_class_counts(labels: np.ndarray, indices: np.ndarray, window: int = 64) -> Dict[str, object]:
    """Summarize class distribution over the full stream and first window."""
    ordered_labels = labels[indices]
    num_classes = int(labels.max()) + 1
    full_counts = {int(c): int((ordered_labels == c).sum()) for c in range(num_classes)}
    head = ordered_labels[: min(window, len(ordered_labels))]
    head_counts = {int(c): int((head == c).sum()) for c in range(num_classes)}
    return {
        "num_samples": int(len(indices)),
        "full_counts": full_counts,
        "head_window": int(len(head)),
        "head_counts": head_counts,
    }


def create_dirichlet_sq_dataloaders(
    data_cfg: DataConfig,
    loader_cfg: LoaderConfig,
    gamma: float,
    seed: int = 0,
    num_slots: Optional[int] = None,
) -> Tuple[Dict[str, DataLoader], int, np.ndarray, Dict[str, object]]:
    """Build SQ loaders with per-domain (per test speed) Dirichlet reorder.

    Matches TRIBE ``LabelDirichletDomainSequence``: each test speed block gets an
    independent Dirichlet label shift, then blocks are concatenated in
    ``data_cfg.test_speeds`` order.
    """
    train_dataset, test_dataset, num_classes = create_sq_datasets(data_cfg)
    labels = np.asarray(test_dataset.labels)
    domain_order = [int(s) for s in data_cfg.test_speeds]
    domain_ids = infer_equal_domain_ids(domain_order, len(labels))
    order, stats = build_dirichlet_domain_sequence(
        labels=labels,
        domain_ids=domain_ids,
        domain_order=domain_order,
        gamma=gamma,
        num_slots=num_slots,
        seed=seed,
        head_window=loader_cfg.batch_size,
    )
    test_subset = Subset(test_dataset, order.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle_train,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    loaders = {"train": train_loader, "test": test_loader}
    return loaders, num_classes, order, stats
