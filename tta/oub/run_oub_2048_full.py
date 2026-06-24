"""Standalone OUB length-2048 trial-split source training and TTA comparison.

Protocol:
- 2048-point non-overlapping windows (~976 per trial).
- Source train: trial1+2; source val: trial3; light subsets use equispaced 100-window subsample.
- Test subsets: light_trial1, light (full-light), full.

Methods:
source_only, bn_adapt, rotta, cotta, tribe, tribe_official, petta, tea, tact, eata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from datasets.oub import OUBDataset
from datasets.paths_oub import DEFAULT_CACHE_DIR, DEFAULT_RAW_DIR
from tta.common import ModelConfig, build_model, evaluate_classification, get_default_device
from tta.dirichlet_imbalance import build_dirichlet_domain_sequence
from tta.oub.oub_2048_data import (
    INPUT_LENGTH,
    NUM_CLASSES,
    NUM_DOMAINS,
    SUBSET_CHOICES,
    get_oub_2048_splits,
)
from tta.oub.task_definitions import OUB_TASKS, parse_tasks_arg
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
from tta.tribe.test_tribe import TRIBE
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


def results_dir_for_subset(subset: str) -> Path:
    return TTA_DIR / "results" / "oub_2048" / subset


def checkpoint_for_domain(
    model: str,
    source_domain: int,
    subset: str = "full",
    light_samples: int = 100,
) -> str:
    stem = f"{model}_oub_B{source_domain + 1}_2048_trial12"
    if subset == "full":
        name = f"{stem}.pth"
    else:
        name = f"{stem}_light{int(light_samples)}.pth"
    return str(ROOT_DIR / "checkpoints" / name)


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
    source_domain: int,
    target_domains: Tuple[int, ...],
    gamma: float,
    args: argparse.Namespace,
) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    x_train, y_train, x_val, y_val, x_test, y_test, meta = get_oub_2048_splits(
        source_domain=source_domain,
        target_domains=target_domains,
        subset=args.subset,
        light_samples=args.light_samples,
        raw_dir=Path(args.raw_dir),
        cache_dir=Path(args.cache_dir),
        force_cache=args.force_cache,
    )
    train_dataset = OUBDataset(x_train, y_train, transform=False, augment=False)
    val_dataset = OUBDataset(x_val, y_val, transform=False, augment=False)
    test_dataset = OUBDataset(x_test, y_test, transform=False, augment=False)
    test_domain_ids = np.asarray(meta["test_domain_ids"], dtype=np.int64)
    order, stats = build_dirichlet_domain_sequence(
        labels=test_dataset.labels,
        domain_ids=test_domain_ids,
        domain_order=target_domains,
        gamma=float(gamma),
        seed=int(args.seed),
        head_window=int(args.batch_size),
    )
    test_subset = Subset(test_dataset, order.tolist())
    stats.update(
        {
            "source_domain": int(source_domain),
            "target_domains": [int(d) for d in target_domains],
            "input_length": INPUT_LENGTH,
            "subset": args.subset,
            "light_samples": int(args.light_samples),
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


def train_source_domain(domain: int, args: argparse.Namespace, device: torch.device) -> Path:
    ckpt_path = Path(checkpoint_for_domain(args.model, domain, args.subset, args.light_samples))
    if ckpt_path.is_file() and not args.force_train:
        print(f"[TRAIN] skip B{domain + 1}, exists: {ckpt_path}")
        return ckpt_path

    set_seed(args.seed)
    loaders, meta = make_loaders(domain, (domain,), gamma=100.0, args=args)
    print(
        f"[TRAIN] B{domain + 1} train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)}"
    )
    model = build_model(model_cfg(None), model_name=args.model, device=device, track_running_stats=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    best_state = None
    best_acc = -1.0
    rows: List[Dict[str, object]] = []

    train_loader = DataLoader(
        loaders["train"].dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )
    val_loader = loaders["val"]

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
            f"[TRAIN] B{domain + 1} epoch={epoch:03d} "
            f"train_acc={train_acc:.2%} val_acc={val_acc:.2%}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model": clone_state_dict(model),
                "epoch": epoch,
                "val_acc": val_acc,
                "source_domain": domain,
                "num_classes": NUM_CLASSES,
                "model_name": args.model,
                "input_length": INPUT_LENGTH,
                "train_trials": [1, 2],
                "val_trial": 3,
                "train_samples": int(meta["data_meta"]["train_samples"]),
                "val_samples": int(meta["data_meta"]["val_samples"]),
            }

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"[TRAIN] saved {ckpt_path}, val_acc={best_acc:.2%}")

    log_dir = results_dir_for_subset(args.subset)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.model}_B{domain + 1}_2048_train_log_seed{args.seed}.csv"
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
    tag = f"{args.model}_oub_2048"
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
        return TRIBE(model=model, device=device, num_classes=NUM_CLASSES, **common)
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


def evaluate_one(
    task_id: str,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    spec = OUB_TASKS[task_id]
    source = int(spec["source"])
    targets = tuple(int(x) for x in spec["targets"])
    gamma = float(spec["gamma"])
    ckpt = args.model_path.strip() or checkpoint_for_domain(
        args.model, source, args.subset, args.light_samples
    )
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Missing source checkpoint for {task_id}: {ckpt}")

    set_seed(args.seed)
    loaders, stats = make_loaders(source, targets, gamma, args)
    test_count = len(loaders["test"].dataset)
    print(f"[EVAL] task={task_id} test_samples={test_count} subset={args.subset}")
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
        f"gamma={gamma},source=B{source + 1},targets={'->'.join(f'B{d + 1}' for d in targets)},"
        f"subset={args.subset},len2048"
    )
    summary = {
        "task": task_id,
        "method": method,
        "protocol": args.protocol,
        "source_only": baseline,
        "acc": acc,
        "delta_source_pp": (acc - baseline) * 100.0,
        "gamma": gamma,
        "source_domain": source,
        "target_domains": list(targets),
        "subset": args.subset,
        "test_samples": test_count,
        "checkpoint": ckpt,
        "extra": extra,
    }
    stamped_batches = [
        {
            "task": task_id,
            "method": method,
            "protocol": args.protocol,
            **row,
            "extra": extra,
        }
        for row in batch_rows
    ]

    out_dir = results_dir_for_subset(args.subset)
    stats_path = out_dir / f"{task_id}_{method}_stream_stats_seed{args.seed}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, stamped_batches


def write_outputs(summary_rows: List[Dict[str, object]], batch_rows: List[Dict[str, object]], args: argparse.Namespace) -> Path:
    out_dir = results_dir_for_subset(args.subset)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = getattr(args, "output_stem", None) or f"oub_2048_{args.subset}_{args.model}_seed{args.seed}"
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
    present = {str(row["method"]) for row in summary_rows}
    methods_order = [m for m in REPORT_METHOD_ORDER if m in present]
    methods_order.extend(sorted(present - set(methods_order)))
    lines = [
        "# OUB Length-2048 TTA Comparison",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| model | {args.model} |",
        f"| input_length | {INPUT_LENGTH} |",
        f"| overlap | 0 (non-overlapping) |",
        f"| windows_per_trial | 976 |",
        f"| train_trials | 1, 2 |",
        f"| val_trial | 3 |",
        f"| train_samples_per_domain | 5×2×{args.light_samples if args.subset != 'full' else 976} = "
        f"{5 * 2 * (args.light_samples if args.subset != 'full' else 976)} |",
        f"| val_samples_per_domain | 5×{args.light_samples if args.subset != 'full' else 976} = "
        f"{5 * (args.light_samples if args.subset != 'full' else 976)} |",
        f"| subset | {args.subset} |",
        f"| light_samples | {args.light_samples} |",
        f"| protocol | {args.protocol} |",
        f"| seed | {args.seed} |",
        "",
        "## Task Results",
        "",
    ]

    header = "| Task | Gamma | " + " | ".join(methods_order) + " | Best TTA |"
    sep = "|---|---:|" + "|".join(["---:"] * len(methods_order)) + "|---:|"
    lines.extend([header, sep])
    for task in tasks:
        sub = [row for row in summary_rows if row["task"] == task]
        by_method = {str(row["method"]): row for row in sub}
        gamma = by_method[methods_order[0]]["gamma"] if methods_order else ""
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
    for domain in range(NUM_DOMAINS):
        train_source_domain(domain, args, device)


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

    write_outputs(summary_rows, batch_rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone OUB len2048 trial-split training + TTA comparison")
    parser.add_argument("command", nargs="?", default="all", choices=["all", "train", "eval"])
    parser.add_argument("--model", type=str, default="tfn", choices=["resnet18", "tfn", "tfn_sttf", "wdcnn"])
    parser.add_argument("--model_path", type=str, default="", help="Override source checkpoint for all eval tasks")
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--protocol", type=str, default="online-batch", choices=sorted(ROTTA_PROTOCOLS))
    parser.add_argument("--subset", type=str, default="light_trial1", choices=list(SUBSET_CHOICES))
    parser.add_argument("--light_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force_cache", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_stem", type=str, default="", help="Result filename stem")

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
    parser.add_argument(
        "--tribe_config",
        type=str,
        default="",
        help="JSON file with official/local TRIBE hyperparameters",
    )
    parser.add_argument(
        "--method_config",
        type=str,
        default="",
        help="Method defaults JSON (e.g. oub_method_defaults_res18.json)",
    )
    parser.add_argument("--tribe_lr", type=float, default=None)
    parser.add_argument("--tribe_optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--tribe_weight_decay", type=float, default=None)
    parser.add_argument("--tribe_steps", type=int, default=None)
    parser.add_argument("--tribe_eta", type=float, default=None)
    parser.add_argument("--tribe_gamma", type=float, default=None)
    parser.add_argument("--tribe_h0", type=float, default=None)
    parser.add_argument("--tribe_lambda", type=float, default=None)
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
            tribe_path = Path(__file__).resolve().parents[1] / "bogie" / "configs" / args.tribe_config
        if not tribe_path.is_file():
            raise FileNotFoundError(f"TRIBE config not found: {args.tribe_config}")
        args.tribe_config_data = load_tribe_config(tribe_path)
        args.tribe_config_path = str(tribe_path)
    config_dir = Path(__file__).resolve().parent / "configs"
    config_path = config_dir / "oub_method_defaults.json"
    if args.method_config:
        config_path = Path(args.method_config)
        if not config_path.is_file():
            config_path = config_dir / args.method_config
        if not config_path.is_file():
            raise FileNotFoundError(f"Method config not found: {args.method_config}")
    elif args.model == "resnet18":
        res18 = config_dir / "oub_method_defaults_res18.json"
        if res18.is_file():
            config_path = res18
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"[INFO] device={device}, subset={args.subset}, light_samples={args.light_samples}")

    if args.command in {"all", "train"} and not args.skip_train:
        run_train(args, device)
    if args.command in {"all", "eval"}:
        run_eval(args, device)


if __name__ == "__main__":
    main()
