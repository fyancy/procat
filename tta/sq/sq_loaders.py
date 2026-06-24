"""Data loaders for SQ tasks T1–T8."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

from tta.common import DataConfig, LoaderConfig, create_sq_dataloaders, parse_speeds_arg
from tta.dirichlet_imbalance import create_dirichlet_sq_dataloaders
from tta.sq.task_definitions import SQ_TASKS


def apply_task_speeds_to_args(args: argparse.Namespace, task_id: str) -> None:
    spec = SQ_TASKS[task_id]
    sources = tuple(int(s) for s in spec["sources"])  # type: ignore[union-attr]
    targets = tuple(int(s) for s in spec["targets"])  # type: ignore[union-attr]
    args.train_speeds = ",".join(str(s) for s in sources)
    if spec["task_type"] == "noise_dyn":
        args.corruption_speeds = ",".join(str(s) for s in targets)
    else:
        args.domain_test_speeds = ",".join(str(s) for s in targets)
    if spec.get("gli_gamma") is not None:
        args.gli_gamma = float(spec["gli_gamma"])  # type: ignore[arg-type]


def make_loaders_for_task(
    task_id: str,
    args: argparse.Namespace,
) -> Tuple[Dict, int, str]:
    spec = SQ_TASKS[task_id]
    task_type = str(spec["task_type"])
    sources = parse_speeds_arg(",".join(str(s) for s in spec["sources"]))  # type: ignore[union-attr]
    targets = parse_speeds_arg(",".join(str(s) for s in spec["targets"]))  # type: ignore[union-attr]
    loader_cfg = LoaderConfig(
        batch_size=args.batch_size,
        shuffle_test=False,
        num_workers=getattr(args, "num_workers", 0),
    )

    if task_type == "cross_domain":
        data_cfg = DataConfig(
            train_ratio=args.train_ratio,
            cross_domain=False,
            transform=False,
            augment_train=False,
            in_channels=1,
            train_speeds=sources,
            test_speeds=targets,
            corruption_type=None,
            severity=0,
        )
        loaders, num_classes = create_sq_dataloaders(data_cfg, loader_cfg)
        return loaders, num_classes, ""

    if task_type == "noise_dyn":
        data_cfg = DataConfig(
            train_ratio=args.train_ratio,
            cross_domain=False,
            transform=False,
            augment_train=False,
            in_channels=1,
            train_speeds=sources,
            test_speeds=targets,
            corruption_type="noise_dyn",
            severity=0,
        )
        loaders, num_classes = create_sq_dataloaders(data_cfg, loader_cfg)
        return loaders, num_classes, "snr=10->0,levels=11"

    if task_type == "gli":
        data_cfg = DataConfig(
            train_ratio=args.train_ratio,
            cross_domain=False,
            transform=False,
            augment_train=False,
            in_channels=1,
            train_speeds=sources,
            test_speeds=targets,
            corruption_type=None,
            severity=0,
        )
        gamma = float(spec.get("gli_gamma") or args.gli_gamma)
        loaders, num_classes, _, _ = create_dirichlet_sq_dataloaders(
            data_cfg=data_cfg,
            loader_cfg=loader_cfg,
            gamma=gamma,
            seed=args.seed,
        )
        return loaders, num_classes, f"gamma={gamma}"

    raise ValueError(f"Unsupported task type: {task_type}")
