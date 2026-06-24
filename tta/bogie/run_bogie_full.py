"""Standalone Bogie multi-speed train/test TTA comparison (3 tasks, chn=0)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
TTA_DIR = ROOT_DIR / "tta"
for _path in (ROOT_DIR, TTA_DIR):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from datasets.bogie import BogieDataset
from tta.bogie.bogie_data import (
    INPUT_LENGTH,
    NUM_CLASSES,
    NUM_SPEEDS,
    SAMPLES_PER_CLASS,
    SPEEDS,
    get_bogie_source_splits,
    get_bogie_splits,
    rpm_for_speed_idx,
)
from tta.bogie.task_definitions import BOGIE_TASKS, SPEED_NAMES, parse_tasks_arg, sources_label
from tta.common import ModelConfig, build_model, evaluate_classification, get_default_device
from tta.petta.petta import PeTTA
from tta.test_bn_adapt import BNAdapt
from tta.test_eata import (
    EATAOfficial,
    collect_bn_affine_params,
    compute_fishers_bn_affine,
    configure_model_for_eata,
)
from tta.test_cotta import CoTTAOfficial, collect_params as collect_cotta_params, configure_model as configure_cotta_model
from tta.test_rotta import ROTTA_PROTOCOLS, RoTTA
from tta.test_tact import build_tact_method
from tta.test_tea import TEAOfficial, collect_params as collect_tea_params, configure_model as configure_tea_model
from tta.tribe.test_tribe import (
    DEFAULT_TRIBE_ETA,
    DEFAULT_TRIBE_GAMMA,
    DEFAULT_TRIBE_H0,
    DEFAULT_TRIBE_LAMBDA,
    DEFAULT_TRIBE_LR,
    DEFAULT_TRIBE_NOISE_STD,
    DEFAULT_TRIBE_STEPS,
    TRIBE,
)
from tta.tribe.tribe_official import TRIBE_OFFICIAL, get_official_defaults

METHODS: Tuple[str, ...] = (
    "source_only",
    "bn_adapt",
    "rotta",
    "cotta",
    "tribe",
    "tribe_official",
    "petta",
    "tea",
    "tact",
    "eata",)

# Markdown/CSV column order (baselines → TRIBE family → Proto variants).
REPORT_METHOD_ORDER: Tuple[str, ...] = (
    "source_only",
    "bn_adapt",
    "rotta",
    "cotta",
    "petta",
    "tea",
    "tact",
    "eata",
    "tribe_official",
    "tribe",)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def results_dir() -> Path:
    return TTA_DIR / "results" / "bogie_2048"


def checkpoint_for_sources(model: str, source_speed_idxs: Sequence[int]) -> str:
    rpms = "_".join(str(rpm_for_speed_idx(int(s))) for s in source_speed_idxs)
    return str(ROOT_DIR / "checkpoints" / f"{model}_bogie_train_rpm{rpms}.pth")


def checkpoint_for_task(model: str, task_id: str) -> str:
    spec = BOGIE_TASKS[task_id]
    return checkpoint_for_sources(model, spec["sources"])


def model_cfg(checkpoint_path: Optional[str]) -> ModelConfig:
    return ModelConfig(
        input_length=INPUT_LENGTH,
        num_classes=NUM_CLASSES,
        transform_in_model=True,
        zero_mean=True,
        in_channels=1,
        checkpoint_path=checkpoint_path,
    )


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_loaders(
    source_speed_idxs: Sequence[int],
    target_speed_idxs: Tuple[int, ...],
    gamma: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    x_train, y_train, x_val, y_val, x_test, y_test, meta = get_bogie_splits(
        source_speed_idxs=source_speed_idxs,
        target_speed_idxs=target_speed_idxs,
        train_ratio=args.train_ratio,
        gamma=float(gamma),
        seed=int(args.seed),
        channel=int(args.channel),
        batch_size=int(args.batch_size),
    )
    train_dataset = BogieDataset(x_train, y_train, transform=False, augment=False)
    val_dataset = BogieDataset(x_val, y_val, transform=False, augment=False)
    test_dataset = BogieDataset(x_test, y_test, transform=False, augment=False)
    order = np.asarray(meta["dirichlet_order"], dtype=np.int64)
    test_subset = Subset(test_dataset, order.tolist())
    stats = dict(meta.get("stream_stats", {}))
    stats.update(
        {
            "source_speed_idxs": [int(s) for s in source_speed_idxs],
            "source_rpms": [rpm_for_speed_idx(int(s)) for s in source_speed_idxs],
            "target_speed_idxs": [int(t) for t in target_speed_idxs],
            "target_rpms": [rpm_for_speed_idx(t) for t in target_speed_idxs],
            "gamma": float(gamma),
            "input_length": INPUT_LENGTH,
            "channel": int(args.channel),
            "data_meta": meta,
        }
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )
    return (
        {
            "train": DataLoader(train_dataset, shuffle=True, **loader_kwargs),
            "val": DataLoader(val_dataset, shuffle=False, **loader_kwargs),
            "test": DataLoader(test_subset, shuffle=False, **loader_kwargs),
        },
        stats,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    optimizer=None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).long()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * int(targets.numel())
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_samples += int(targets.numel())
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_task(task_id: str, args: argparse.Namespace, device: torch.device) -> Path:
    spec = BOGIE_TASKS[task_id]
    sources = [int(s) for s in spec["sources"]]
    ckpt_path = Path(checkpoint_for_task(args.model, task_id))
    src_label = sources_label(sources)
    if ckpt_path.is_file() and not args.force_train:
        print(f"[TRAIN] skip {task_id} ({src_label}), exists: {ckpt_path}")
        return ckpt_path

    set_seed(args.seed)
    x_train, y_train, x_val, y_val, meta = get_bogie_source_splits(
        source_speed_idxs=sources,
        train_ratio=args.train_ratio,
        seed=args.seed,
        channel=args.channel,
    )
    train_dataset = BogieDataset(x_train, y_train, transform=False, augment=False)
    val_dataset = BogieDataset(x_val, y_val, transform=False, augment=False)
    print(f"[TRAIN] {task_id} sources={src_label} train={len(train_dataset)} val={len(val_dataset)}")
    model = build_model(model_cfg(None), model_name=args.model, device=device, track_running_stats=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    best_state = None
    best_acc = -1.0
    rows: List[Dict[str, object]] = []

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"[TRAIN] {task_id} epoch={epoch:03d} train_acc={train_acc:.2%} val_acc={val_acc:.2%}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model": clone_state_dict(model),
                "epoch": epoch,
                "val_acc": val_acc,
                "task_id": task_id,
                "source_speed_idxs": sources,
                "source_rpms": [rpm_for_speed_idx(s) for s in sources],
                "num_classes": NUM_CLASSES,
                "model_name": args.model,
                "input_length": INPUT_LENGTH,
                "channel": int(args.channel),
                "train_samples": int(meta["train_samples"]),
                "val_samples": int(meta["val_samples"]),
            }

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"[TRAIN] saved {ckpt_path}, val_acc={best_acc:.2%}")

    log_dir = results_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.model}_{task_id}_train_log_seed{args.seed}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return ckpt_path


def maybe_alias_classifier_for_tact(model: nn.Module) -> nn.Module:
    if not hasattr(model, "fc") and hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
        model.fc = model.backbone.fc
    return model


def load_tribe_config(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("tribe", data)
    return dict(block)


def tribe_official_params(args: argparse.Namespace) -> Dict[str, object]:
    base = get_official_defaults()
    if getattr(args, "tribe_config_data", None):
        base.update(args.tribe_config_data)

    def pick(name: str, arg_name: str, cast, fallback):
        cli = getattr(args, arg_name, None)
        if cli is not None:
            return cast(cli)
        if name in base:
            return cast(base[name])
        return cast(fallback)

    return {
        "lr": pick("lr", "tribe_lr", float, 1e-3),
        "optimizer": str(getattr(args, "tribe_optimizer", None) or base.get("optimizer", "adam")),
        "weight_decay": pick("weight_decay", "tribe_weight_decay", float, 0.0),
        "steps": pick("steps", "tribe_steps", int, 1),
        "eta": pick("eta", "tribe_eta", float, 0.01),
        "gamma": pick("gamma", "tribe_gamma", float, 0.0),
        "h0": pick("h0", "tribe_h0", float, 0.05),
        "lambda_reg": pick("lambda_reg", "tribe_lambda", float, 0.5),
        "gaussian_std": pick("gaussian_std", "tribe_gaussian_std", float, 0.005),
    }


def tribe_params(args: argparse.Namespace) -> Dict[str, object]:
    base = {
        "lr": DEFAULT_TRIBE_LR,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "steps": DEFAULT_TRIBE_STEPS,
        "eta": DEFAULT_TRIBE_ETA,
        "gamma": DEFAULT_TRIBE_GAMMA,
        "h0": DEFAULT_TRIBE_H0,
        "lambda_reg": DEFAULT_TRIBE_LAMBDA,
        "adapt_params": "affine",
        "noise_std": DEFAULT_TRIBE_NOISE_STD,
    }
    if getattr(args, "tribe_config_data", None):
        base.update(args.tribe_config_data)

    def pick(name: str, arg_name: str, cast):
        cli = getattr(args, arg_name, None)
        if cli is not None:
            return cast(cli)
        return cast(base[name])

    return {
        "lr": pick("lr", "tribe_lr", float),
        "optimizer": str(pick("optimizer", "tribe_optimizer", str)),
        "weight_decay": pick("weight_decay", "tribe_weight_decay", float),
        "steps": pick("steps", "tribe_steps", int),
        "eta": pick("eta", "tribe_eta", float),
        "gamma": pick("gamma", "tribe_gamma", float),
        "h0": pick("h0", "tribe_h0", float),
        "lambda_reg": pick("lambda_reg", "tribe_lambda", float),
        "adapt_params": str(pick("adapt_params", "tribe_adapt_params", str)),
        "noise_std": pick("noise_std", "tribe_noise_std", float),
    }


def build_adapter(
    method: str,
    cfg: ModelConfig,
    loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    task_id: str = "",
) -> object:
    model = build_model(cfg, model_name=args.model, device=device, track_running_stats=True)
    common = dict(protocol=args.protocol, online_batch_size=args.batch_size)
    tag = f"{args.model}_bogie_2048"
    if method == "bn_adapt":
        return BNAdapt(model, device=device, momentum=args.bn_momentum)
    if method == "rotta":
        return RoTTA(model=model, device=device, num_classes=NUM_CLASSES, **common)
    if method == "cotta":
        model = configure_cotta_model(model)
        params, _ = collect_cotta_params(model)
        if not params:
            raise RuntimeError("CoTTA selected no trainable parameters.")
        optimizer = optim.Adam(params, lr=args.cotta_lr, betas=(0.9, 0.999))
        return CoTTAOfficial(
            model=model,
            optimizer=optimizer,
            device=device,
            steps=args.cotta_steps,
            episodic=args.cotta_episodic,
            mt_alpha=args.cotta_mt_alpha,
            rst_m=args.cotta_rst_m,
            ap=args.cotta_ap,
            N=args.cotta_N,
            gaussian_std=args.cotta_gaussian_std,
        )
    if method == "tribe":
        tp = tribe_params(args)
        return TRIBE(
            model=model,
            device=device,
            num_classes=NUM_CLASSES,
            lr=float(tp["lr"]),
            optimizer_name=str(tp["optimizer"]),
            weight_decay=float(tp["weight_decay"]),
            steps=int(tp["steps"]),
            eta=float(tp["eta"]),
            gamma=float(tp["gamma"]),
            lambda_reg=float(tp["lambda_reg"]),
            h0=float(tp["h0"]),
            adapt_params=str(tp["adapt_params"]),
            noise_std=float(tp["noise_std"]),
            **common,
        )
    if method == "tribe_official":
        op = tribe_official_params(args)
        return TRIBE_OFFICIAL(
            model=model,
            device=device,
            num_classes=NUM_CLASSES,
            lr=float(op["lr"]),
            optimizer_name=str(op["optimizer"]),
            weight_decay=float(op["weight_decay"]),
            steps=int(op["steps"]),
            eta=float(op["eta"]),
            gamma=float(op["gamma"]),
            lambda_reg=float(op["lambda_reg"]),
            h0=float(op["h0"]),
            gaussian_std=float(op["gaussian_std"]),
            **common,
        )
    if method == "petta":
        return PeTTA(
            model=model,
            device=device,
            num_classes=NUM_CLASSES,
            source_loader=loaders["train"],
            proto_cache_tag=tag,
            **common,
        )
    if method == "tea":
        model = configure_tea_model(model, adapt_params=args.tea_adapt_params)
        params, _ = collect_tea_params(model, adapt_params=args.tea_adapt_params)
        if not params:
            raise RuntimeError("TEA selected no trainable parameters.")
        optimizer = (
            optim.Adam(params, lr=args.tea_lr)
            if args.tea_optimizer == "adam"
            else optim.SGD(params, lr=args.tea_lr)
        )
        return TEAOfficial(
            model=model,
            device=device,
            optimizer=optimizer,
            steps=args.tea_steps,
            sgld_steps=args.tea_sgld_steps,
            sgld_lr=args.tea_sgld_lr,
            sgld_std=args.tea_sgld_std,
            reinit_freq=args.tea_reinit_freq,
            n_classes=NUM_CLASSES,
            im_sz=INPUT_LENGTH,
            n_ch=1,
            adapt_params=args.tea_adapt_params,
        )
    if method == "tact":
        model = maybe_alias_classifier_for_tact(model)
        return build_tact_method(
            model=model,
            device=device,
            num_classes=NUM_CLASSES,
            use_adapt=args.tact_use_adapt,
            num_aug=args.tact_num_aug,
            start_pc=args.tact_start_pc,
            num_pcs=args.tact_num_pcs,
            adaptation_lr=args.tact_lr,
            entropy_weighting=args.tact_entropy_weighting,
            noise_std=args.tact_noise_std,
        )
    if method == "eata":
        model = configure_model_for_eata(model)
        params, _ = collect_bn_affine_params(model)
        if not params:
            raise RuntimeError("EATA selected no trainable parameters.")
        optimizer = optim.Adam(params, lr=args.eata_lr, betas=(0.9, 0.999))
        fishers = None
        if args.eata_use_fisher:
            fisher_model = build_model(cfg, model_name=args.model, device=device, track_running_stats=True)
            fishers = compute_fishers_bn_affine(
                fisher_model, loaders["train"], device, max_batches=args.eata_fisher_batches
            )
        e_margin = args.eata_e_margin if args.eata_e_margin is not None else 0.5 * math.log(NUM_CLASSES)
        return EATAOfficial(
            model=model,
            optimizer=optimizer,
            device=device,
            fishers=fishers,
            fisher_alpha=args.eata_fisher_alpha,
            steps=args.eata_steps,
            e_margin=e_margin,
            d_margin=args.eata_d_margin,
        )
    raise ValueError(f"Unsupported method: {method}")


def evaluate_adapter_stream(
    adapter,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, List[Dict[str, object]]]:
    total_correct = 0
    total_samples = 0
    rows: List[Dict[str, object]] = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        logits = adapter.adapt_one_batch(inputs)
        preds = logits.argmax(dim=1)
        correct = int((preds == labels).sum().item())
        samples = int(labels.numel())
        total_correct += correct
        total_samples += samples
        rows.append(
            {
                "batch_idx": batch_idx,
                "batch_size": samples,
                "batch_correct": correct,
                "batch_acc": correct / max(samples, 1),
                "cumulative_acc": total_correct / max(total_samples, 1),
            }
        )
        if max_batches and (batch_idx + 1) >= max_batches:
            break
    return total_correct / max(total_samples, 1), rows


def _target_label(targets: Tuple[int, ...]) -> str:
    return "->".join(SPEED_NAMES[t] for t in targets)


def _json_safe(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def evaluate_one(
    task_id: str,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    spec = BOGIE_TASKS[task_id]
    sources = tuple(int(x) for x in spec["sources"])
    targets = tuple(int(x) for x in spec["targets"])
    gamma = float(spec["gamma"])
    ckpt = args.model_path.strip() or checkpoint_for_task(args.model, task_id)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Missing source checkpoint for {task_id}: {ckpt}")

    set_seed(args.seed)
    loaders, stats = make_loaders(sources, targets, gamma, args)
    test_count = len(loaders["test"].dataset)
    print(f"[EVAL] task={task_id} test_samples={test_count}")
    cfg = model_cfg(ckpt)
    baseline_model = build_model(cfg, model_name=args.model, device=device, track_running_stats=True)
    baseline = float(
        evaluate_classification(
            baseline_model,
            loaders["test"],
            device=device,
            criterion=nn.CrossEntropyLoss(),
            max_batches=args.max_batches or None,
        ).get("acc", 0.0)
    )

    if method == "source_only":
        acc = baseline
        batch_rows: List[Dict[str, object]] = []
    else:
        adapter = build_adapter(method, cfg, loaders, args, device, task_id=task_id)
        acc, batch_rows = evaluate_adapter_stream(adapter, loaders["test"], device, max_batches=args.max_batches)

    extra = (
        f"gamma={gamma},sources={sources_label(sources)},targets={_target_label(targets)},"
        f"chn={args.channel},len2048"
    )
    summary = {
        "task": task_id,
        "method": method,
        "protocol": args.protocol,
        "source_only": baseline,
        "acc": acc,
        "delta_source_pp": (acc - baseline) * 100.0,
        "gamma": gamma,
        "source_speed_idxs": list(sources),
        "source_rpms": [rpm_for_speed_idx(s) for s in sources],
        "target_speed_idxs": list(targets),
        "test_samples": test_count,
        "checkpoint": ckpt,
        "extra": extra,
    }
    stamped_batches = [
        {"task": task_id, "method": method, "protocol": args.protocol, **row, "extra": extra}
        for row in batch_rows
    ]

    out_dir = results_dir()
    stats_path = out_dir / f"{task_id}_{method}_stream_stats_seed{args.seed}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(_json_safe(stats), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary, stamped_batches


def merge_tribe_official_rows(
    summary_rows: List[Dict[str, object]],
    out_dir: Path,
    model: str,
    seed: int,
) -> List[Dict[str, object]]:
    """Replace tribe_official rows in main summary with latest faithful-port eval."""
    supp_path = out_dir / f"bogie_2048_{model}_seed{seed}_tribe_official.csv"
    if not supp_path.is_file():
        return summary_rows

    supplemental = list(csv.DictReader(supp_path.open(encoding="utf-8")))
    if not supplemental:
        return summary_rows

    kept = [row for row in summary_rows if str(row.get("method")) != "tribe_official"]
    official_rows = [row for row in supplemental if str(row.get("method")) == "tribe_official"]
    if not official_rows:
        official_rows = [{**row, "method": "tribe_official"} for row in supplemental]
    return kept + official_rows


def report_stem(args: argparse.Namespace) -> str:
    return getattr(args, "output_stem", None) or f"bogie_2048_{args.model}_seed{args.seed}"


def sync_partial_methods_to_main(
    args: argparse.Namespace,
    new_rows: List[Dict[str, object]],
    methods_updated: Sequence[str],
) -> None:
    """Replace selected method rows in an existing comparison table (keep other methods)."""
    out_dir = results_dir()
    main_stem = report_stem(args)
    main_csv = out_dir / f"{main_stem}.csv"
    if not main_csv.is_file():
        write_outputs(new_rows, [], args)
        return

    existing = list(csv.DictReader(main_csv.open(encoding="utf-8")))
    new_keys = {(str(row["task"]), str(row["method"])) for row in new_rows}
    kept = [
        row
        for row in existing
        if (str(row.get("task")), str(row.get("method"))) not in new_keys
    ]
    merged = kept + [{k: row[k] for k in row} for row in new_rows]
    for row in merged:
        for key in ("source_only", "acc", "delta_source_pp", "gamma", "test_samples"):
            if key in row and row[key] not in ("", None):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
    write_outputs(merged, [], args)
    print(f"[MERGE] Updated {len(new_rows)} row(s) in {main_csv}")


def sync_tribe_official_main_report(
    args: argparse.Namespace,
    official_rows: List[Dict[str, object]],
) -> None:
    """After tribe_official-only eval, refresh the main comparison csv/md."""
    out_dir = results_dir()
    main_stem = f"bogie_2048_{args.model}_seed{args.seed}"
    main_csv = out_dir / f"{main_stem}.csv"
    if not main_csv.is_file():
        write_outputs(official_rows, [], args)
        return

    existing = list(csv.DictReader(main_csv.open(encoding="utf-8")))
    new_keys = {(str(row["task"]), str(row["method"])) for row in official_rows}
    kept = [
        row
        for row in existing
        if (str(row.get("task")), str(row.get("method"))) not in new_keys
    ]
    merged = kept + [{k: row[k] for k in row} for row in official_rows]
    for row in merged:
        for key in ("source_only", "acc", "delta_source_pp", "gamma", "test_samples"):
            if key in row and row[key] not in ("", None):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
    sync_args = argparse.Namespace(**{**vars(args), "output_stem": ""})
    write_outputs(merged, [], sync_args)


def report_methods_order(summary_rows: List[Dict[str, object]]) -> List[str]:
    present = {str(row["method"]) for row in summary_rows}
    ordered = [m for m in REPORT_METHOD_ORDER if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def write_outputs(
    summary_rows: List[Dict[str, object]], batch_rows: List[Dict[str, object]], args: argparse.Namespace
) -> Path:
    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = getattr(args, "output_stem", None) or f"bogie_2048_{args.model}_seed{args.seed}"
    default_stem = f"bogie_2048_{args.model}_seed{args.seed}"
    if stem == default_stem:
        summary_rows = merge_tribe_official_rows(summary_rows, out_dir, args.model, args.seed)
    summary_csv = out_dir / f"{stem}.csv"
    batch_csv = out_dir / f"{stem}_batch_accuracy.csv"
    md_path = out_dir / f"{stem}.md"

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if batch_rows:
        fields: List[str] = []
        for row in batch_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with batch_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(batch_rows)

    tasks = sorted({str(row["task"]) for row in summary_rows})
    methods_order = report_methods_order(summary_rows)
    has_tribe_official = "tribe_official" in methods_order
    lines = [
        "# Bogie Cross-Speed TTA Comparison",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| model | {args.model} |",
        f"| input_length | {INPUT_LENGTH} |",
        f"| channel | {args.channel} (FSx) |",
        f"| num_classes | {NUM_CLASSES} |",
        f"| samples_per_class | {SAMPLES_PER_CLASS} |",
        f"| train_ratio | {args.train_ratio} |",
        f"| protocol | {args.protocol} |",
        f"| seed | {args.seed} |",
    ]
    if has_tribe_official:
        lines.append(
            "| tribe_official | "
            "faithful port: official BN/aug/loss + `tribe_config_official.json` |"
        )
        lines.append("| tribe | local port: RoTTA strong aug, local BN, lambda=0, eta=0.1 |")
    if getattr(args, "tribe_config_path", None):
        lines.append(f"| tribe_config | {args.tribe_config_path} |")
    lines.extend(
        [
            "",
            "## Task Results",
            "",
        ]
    )

    header = "| Task | Gamma | " + " | ".join(methods_order) + " | Best TTA |"
    sep = "|---|---:|" + "|".join(["---:"] * len(methods_order)) + "|---:|"
    lines.extend([header, sep])
    for task in tasks:
        sub = [row for row in summary_rows if row["task"] == task]
        by_method = {str(row["method"]): row for row in sub}
        gamma = next((row["gamma"] for row in sub), "")
        cells = []
        for method in methods_order:
            row = by_method.get(method)
            cells.append("-" if row is None else f"{float(row['acc']):.2%} ({float(row['delta_source_pp']):+.1f})")
        other_rows = [row for row in sub if row["method"] != "source_only"]
        best_other = max(other_rows, key=lambda row: float(row["acc"])) if other_rows else None
        best_label = "-" if best_other is None else f"{best_other['method']} {float(best_other['acc']):.2%}"
        lines.append(f"| {task} | {gamma} | " + " | ".join(cells) + f" | {best_label} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OUT] {summary_csv}")
    if batch_rows:
        print(f"[OUT] {batch_csv}")
    print(f"[OUT] {md_path}")
    return md_path


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    tasks = parse_tasks_arg(args.tasks)
    seen: set[str] = set()
    for task_id in tasks:
        key = checkpoint_for_task(args.model, task_id)
        if key in seen:
            continue
        seen.add(key)
        train_task(task_id, args, device)


def run_eval(args: argparse.Namespace, device: torch.device) -> None:
    tasks = parse_tasks_arg(args.tasks)
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {METHODS}")

    summary_rows: List[Dict[str, object]] = []
    batch_rows: List[Dict[str, object]] = []
    for task in tasks:
        for method in methods:
            print(f"[EVAL] task={task} method={method}")
            summary, batches = evaluate_one(task, method, args, device)
            summary_rows.append(summary)
            batch_rows.extend(batches)
            print(
                f"[EVAL] task={task} method={method} "
                f"source={float(summary['source_only']):.2%} acc={float(summary['acc']):.2%}"
            )

    main_csv = results_dir() / f"{report_stem(args)}.csv"
    stem = getattr(args, "output_stem", None) or ""
    if methods == ["tribe_official"] and stem.endswith("_tribe_official"):
        write_outputs(summary_rows, batch_rows, args)
        sync_tribe_official_main_report(args, summary_rows)
    elif len(methods) < len(METHODS) and main_csv.is_file():
        sync_partial_methods_to_main(args, summary_rows, methods)
    else:
        write_outputs(summary_rows, batch_rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bogie cross-speed training + TTA comparison")
    parser.add_argument("command", nargs="?", default="all", choices=["all", "train", "eval"])
    parser.add_argument("--model", type=str, default="tfn", choices=["resnet18", "tfn", "tfn_sttf", "wdcnn"])
    parser.add_argument("--model_path", type=str, default="", help="Override source checkpoint for all eval tasks")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--protocol", type=str, default="online-batch", choices=sorted(ROTTA_PROTOCOLS))
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)

    parser.add_argument("--bn_momentum", type=float, default=1.0)
    parser.add_argument("--tea_lr", type=float, default=1e-3)
    parser.add_argument("--tea_optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--tea_steps", type=int, default=1)
    parser.add_argument("--tea_sgld_steps", type=int, default=1)
    parser.add_argument("--tea_sgld_lr", type=float, default=1.0)
    parser.add_argument("--tea_sgld_std", type=float, default=0.01)
    parser.add_argument("--tea_reinit_freq", type=float, default=0.05)
    parser.add_argument("--tea_adapt_params", type=str, default="bn", choices=["bn", "affine", "all"])
    parser.add_argument("--tact_use_adapt", action="store_true")
    parser.add_argument("--tact_num_aug", type=int, default=8)
    parser.add_argument("--tact_start_pc", type=int, default=0)
    parser.add_argument("--tact_num_pcs", type=int, default=1)
    parser.add_argument("--tact_lr", type=float, default=1e-4)
    parser.add_argument("--tact_entropy_weighting", type=float, default=10.0)
    parser.add_argument("--tact_noise_std", type=float, default=0.05)
    parser.add_argument("--eata_lr", type=float, default=1e-3)
    parser.add_argument("--eata_steps", type=int, default=1)
    parser.add_argument("--eata_e_margin", type=float, default=None)
    parser.add_argument("--eata_d_margin", type=float, default=0.05)
    parser.add_argument("--eata_use_fisher", action="store_true")
    parser.add_argument("--eata_fisher_alpha", type=float, default=2000.0)
    parser.add_argument("--eata_fisher_batches", type=int, default=10)
    parser.add_argument("--cotta_lr", type=float, default=1e-3)
    parser.add_argument("--cotta_steps", type=int, default=1)
    parser.add_argument("--cotta_episodic", action="store_true")
    parser.add_argument("--cotta_mt_alpha", type=float, default=0.999)
    parser.add_argument("--cotta_rst_m", type=float, default=0.01)
    parser.add_argument("--cotta_ap", type=float, default=0.92)
    parser.add_argument("--cotta_N", type=int, default=32)
    parser.add_argument("--cotta_gaussian_std", type=float, default=0.005)
    parser.add_argument("--tribe_config", type=str, default="", help="JSON file with official/local TRIBE hyperparameters")
    parser.add_argument("--method_config", type=str, default="", help="Method defaults JSON (e.g. bogie_method_defaults_res18.json)")
    parser.add_argument("--output_stem", type=str, default="", help="Result filename stem (avoid overwriting default report)")
    parser.add_argument("--tribe_lr", type=float, default=None)
    parser.add_argument("--tribe_optimizer", type=str, default=None, choices=["adam", "sgd"])
    parser.add_argument("--tribe_weight_decay", type=float, default=None)
    parser.add_argument("--tribe_steps", type=int, default=None)
    parser.add_argument("--tribe_eta", type=float, default=None)
    parser.add_argument("--tribe_gamma", type=float, default=None)
    parser.add_argument("--tribe_h0", type=float, default=None)
    parser.add_argument("--tribe_lambda", type=float, default=None)
    parser.add_argument("--tribe_adapt_params", type=str, default=None, choices=["affine", "bias"])
    parser.add_argument("--tribe_noise_std", type=float, default=None)
    parser.add_argument("--tribe_gaussian_std", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tribe_config_data = {}
    args.tribe_config_path = ""
    if args.tribe_config:
        tribe_path = Path(args.tribe_config)
        if not tribe_path.is_file():
            tribe_path = Path(__file__).resolve().parent / "configs" / args.tribe_config
        if not tribe_path.is_file():
            raise FileNotFoundError(f"TRIBE config not found: {args.tribe_config}")
        args.tribe_config_data = load_tribe_config(tribe_path)
        args.tribe_config_path = str(tribe_path)
    config_dir = Path(__file__).resolve().parent / "configs"
    config_path = config_dir / "bogie_method_defaults.json"
    if args.method_config:
        config_path = Path(args.method_config)
        if not config_path.is_file():
            config_path = config_dir / args.method_config
        if not config_path.is_file():
            raise FileNotFoundError(f"Method config not found: {args.method_config}")
    elif args.model == "resnet18":
        res18 = config_dir / "bogie_method_defaults_res18.json"
        if res18.is_file():
            config_path = res18
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"[INFO] device={device}, channel={args.channel}, train_ratio={args.train_ratio}")

    if args.command in {"all", "train"} and not args.skip_train:
        run_train(args, device)
    if args.command in {"all", "eval"}:
        run_eval(args, device)


if __name__ == "__main__":
    main()
