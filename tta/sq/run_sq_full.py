"""SQ eight-task TTA comparison (Bogie-aligned 12 methods, online-batch).

Tasks: T1–T6 cross-domain (2 source speeds → 2 target speeds), T7 noise_dyn, T8 gli.
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

ROOT_DIR = Path(__file__).resolve().parents[2]
TTA_DIR = ROOT_DIR / "tta"
for _path in (ROOT_DIR, TTA_DIR):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from torch.utils.data import DataLoader

from datasets.sq import SQDataset, get_sq_data
from tta.common import ModelConfig, build_model, evaluate_classification, get_default_device, parse_speeds_arg
from tta.petta.petta import PeTTA
from tta.test_bn_adapt import BNAdapt
from tta.test_cotta import CoTTAOfficial, collect_params as collect_cotta_params, configure_model as configure_cotta_model
from tta.test_eata import (
    EATAOfficial,
    collect_bn_affine_params,
    compute_fishers_bn_affine,
    configure_model_for_eata,
)
from tta.test_rotta import ROTTA_PROTOCOLS, RoTTA
from tta.test_tact import build_tact_method
from tta.test_tea import TEAOfficial, collect_params as collect_tea_params, configure_model as configure_tea_model
from tta.tribe.test_tribe import TRIBE
from tta.tribe.tribe_official import TRIBE_OFFICIAL, get_official_defaults

from tta.sq.scenario_definitions import (
    SQ_INPUT_LENGTH,
    SQ_NUM_CLASSES,
    load_method_config_json,
)
from tta.sq.sq_loaders import make_loaders_for_task
from tta.sq.task_definitions import (
    LEGACY_SCENARIO_TO_TASK,
    SQ_TASKS,
    SQ_TASK_ORDER,
    checkpoint_for_task,
    format_task_definitions_table,
    parse_tasks_arg,
    proto_scenario_label,
    sources_tag,
    task_by_result_key,
    task_result_key,
)

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

SCENARIO_ORDER = SQ_TASK_ORDER

MODEL_DEFAULTS = {
    "resnet18": "checkpoints/resnet18_sq_clean_noaug.pth",
    "tfn": "checkpoints/tfn_sq_clean_noaug.pth",
    "tfn_sttf": "checkpoints/tfn_sq_clean_noaug.pth",
    "wdcnn": "checkpoints/wdcnn_sq_clean_noaug_bs32_e120.pth",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def results_dir() -> Path:
    return TTA_DIR / "results" / "sq_bogie12"


def default_checkpoint(model: str) -> str:
    rel = MODEL_DEFAULTS.get(model, MODEL_DEFAULTS["tfn"])
    return str(ROOT_DIR / rel)


def bn_momentum_for_task(task_id: str) -> float:
    if task_id in ("T7",):
        return 0.5
    return 1.0


def bn_momentum_for(scenario: str) -> float:
    """Backward-compatible alias."""
    if scenario in ("noise_dyn", "T7"):
        return 0.5
    return 1.0


def model_cfg(checkpoint_path: str, num_classes: int) -> ModelConfig:
    return ModelConfig(
        input_length=2048,
        num_classes=num_classes,
        transform_in_model=True,
        zero_mean=True,
        in_channels=1,
        checkpoint_path=checkpoint_path,
    )


def maybe_alias_classifier_for_tact(model: nn.Module) -> nn.Module:
    if not hasattr(model, "fc") and hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
        model.fc = model.backbone.fc
    return model


def load_tribe_config(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("tribe", data))


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
    loaders: Dict,
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
    scenario: str,
    scenario_label: str = "",
) -> object:
    model = build_model(cfg, model_name=args.model, device=device, track_running_stats=True)
    common = dict(protocol=args.protocol, online_batch_size=args.batch_size)
    tag = f"{args.model}_sq"
    if method == "bn_adapt":
        return BNAdapt(model, device=device, momentum=bn_momentum_for_task(scenario))
    if method == "rotta":
        return RoTTA(model=model, device=device, num_classes=num_classes, **common)
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
        return TRIBE(model=model, device=device, num_classes=num_classes, **common)
    if method == "tribe_official":
        op = tribe_official_params(args)
        return TRIBE_OFFICIAL(
            model=model,
            device=device,
            num_classes=num_classes,
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
            num_classes=num_classes,
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
            n_classes=num_classes,
            im_sz=2048,
            n_ch=1,
            adapt_params=args.tea_adapt_params,
        )
    if method == "tact":
        model = maybe_alias_classifier_for_tact(model)
        return build_tact_method(
            model=model,
            device=device,
            num_classes=num_classes,
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
        e_margin = args.eata_e_margin if args.eata_e_margin is not None else 0.5 * math.log(num_classes)
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


def parse_scenarios(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return ["domain", "noise_dyn", "gli"]
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_tasks(args: argparse.Namespace) -> List[str]:
    tasks_raw = getattr(args, "tasks", "") or ""
    if tasks_raw.strip():
        return parse_tasks_arg(tasks_raw)
    legacy = parse_scenarios(args.scenarios)
    out: List[str] = []
    for item in legacy:
        mapped = LEGACY_SCENARIO_TO_TASK.get(item, item)
        if mapped in SQ_TASK_ORDER:
            out.append(mapped)
        elif item in SQ_TASK_ORDER:
            out.append(item)
        else:
            raise ValueError(f"Unknown scenario/task: {item}")
    return out


def evaluate_one(
    task_id: str,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    ckpt = args.model_path.strip() or checkpoint_for_task(args.model, task_id)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Missing checkpoint for {task_id}: {ckpt}")

    set_seed(args.seed)
    loaders, num_classes, extra = make_loaders_for_task(task_id, args)
    label = task_result_key(task_id, extra)
    proto_label = proto_scenario_label(task_id, extra)
    cfg = model_cfg(ckpt, num_classes)
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
        adapter = build_adapter(
            method, cfg, loaders, args, device, num_classes, task_id, scenario_label=proto_label
        )
        acc, batch_rows = evaluate_stream(
            scenario=label,
            data_loader=loaders["test"],
            device=device,
            logits_fn=adapter.adapt_one_batch,
            extra=extra,
            max_batches=args.max_batches,
        )
        for row in batch_rows:
            row["method"] = method
            row["task"] = label

    summary = {
        "task": label,
        "scenario": label,
        "method": method,
        "protocol": args.protocol,
        "source_only": baseline,
        "acc": acc,
        "delta_source_pp": (acc - baseline) * 100.0,
        "model": args.model,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "checkpoint": ckpt,
        "extra": extra,
    }
    stamped = [
        {"task": label, "scenario": label, "method": method, "protocol": args.protocol, **row, "extra": extra}
        for row in batch_rows
    ]
    return summary, stamped


def resolve_method_config_path(args: argparse.Namespace) -> Path:
    config_dir = Path(__file__).resolve().parent / "configs"
    if args.method_config:
        candidate = Path(args.method_config)
        if candidate.is_file():
            return candidate
        candidate = config_dir / args.method_config
        if candidate.is_file():
            return candidate
    model = str(getattr(args, "model", "tfn"))
    if model == "resnet18":
        res18 = config_dir / "sq_method_defaults_res18.json"
        if res18.is_file():
            return res18
    return config_dir / "sq_method_defaults.json"


def build_sq_eval_markdown(
    summary_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    methods_order: Optional[List[str]] = None,
) -> List[str]:
    present = {str(row["method"]) for row in summary_rows}
    if methods_order is None:
        methods_order = [m for m in REPORT_METHOD_ORDER if m in present]
        methods_order.extend(sorted(present - set(methods_order)))

    scenarios = [s for s in SCENARIO_ORDER if any(str(r.get("task", r.get("scenario"))) == s for r in summary_rows)]
    scenarios.extend(
        sorted({str(r.get("task", r.get("scenario"))) for r in summary_rows} - set(scenarios))
    )

    config_path = getattr(args, "method_config_path", None) or resolve_method_config_path(args)
    cfg_json = getattr(args, "method_config_data", None) or load_method_config_json(config_path)

    lines = [
        "# SQ Eight-Task TTA Comparison (10 baseline methods)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        "| dataset | SQ |",
        f"| model | {args.model} |",
        f"| input_length | {SQ_INPUT_LENGTH} |",
        f"| num_classes | {SQ_NUM_CLASSES} |",
        f"| train_ratio | {args.train_ratio} |",
        f"| protocol | {args.protocol} |",
        f"| batch_size | {args.batch_size} |",
        f"| seed | {args.seed} |",
        f"| method_config | `{config_path}` |",
        f"| methods | 10 ({', '.join(methods_order)}) |",
        "",
        "## Task Definitions (T1–T8)",
        "",
    ]
    lines.extend(format_task_definitions_table())
    lines.extend(
        [
            "",
            "## Task Results",
            "",
        ]
    )
    header = "| Task | " + " | ".join(methods_order) + " | Best TTA |"
    sep = "|---|" + "|".join(["---:"] * len(methods_order)) + "|---:|"
    lines.extend([header, sep])
    for task_key in scenarios:
        spec = task_by_result_key(task_key)
        task_label = task_key
        if spec:
            task_label = f"{task_key} ({spec['label_zh']})"
        sub = [row for row in summary_rows if str(row.get("task", row.get("scenario"))) == task_key]
        by_method = {str(row["method"]): row for row in sub}
        cells = []
        for method in methods_order:
            row = by_method.get(method)
            cells.append("-" if row is None else f"{float(row['acc']):.2%} ({float(row['delta_source_pp']):+.1f})")
        other_rows = [row for row in sub if row["method"] != "source_only"]
        best_other = max(other_rows, key=lambda row: float(row['acc'])) if other_rows else None
        best_text = (
            "-"
            if best_other is None
            else f"{best_other['method']} {float(best_other['acc']):.2%}"
        )
        lines.append(f"| {task_label} | " + " | ".join(cells) + f" | {best_text} |")
    return lines


def write_outputs(summary_rows: List[Dict[str, object]], batch_rows: List[Dict[str, object]], args: argparse.Namespace) -> Path:
    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = getattr(args, "output_stem", None) or f"sq_bogie12_{args.model}_seed{args.seed}"
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

    lines = build_sq_eval_markdown(summary_rows, args)

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OUT] {summary_csv}")
    print(f"[OUT] {batch_csv if batch_rows else '(no batch rows)'}")
    print(f"[OUT] {md_path}")
    return summary_csv


def run_eval(args: argparse.Namespace, device: torch.device) -> None:
    tasks = resolve_tasks(args)
    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Supported: {METHODS}")

    summary_rows: List[Dict[str, object]] = []
    batch_rows: List[Dict[str, object]] = []
    for task_id in tasks:
        for method in methods:
            print(f"[EVAL] task={task_id} method={method}")
            summary, batches = evaluate_one(task_id, method, args, device)
            summary_rows.append(summary)
            batch_rows.extend(batches)
            print(
                f"[EVAL] task={task_id} method={method} "
                f"source={float(summary['source_only']):.2%} acc={float(summary['acc']):.2%}"
            )
    if getattr(args, "merge_with_existing", False):
        stem = getattr(args, "output_stem", None) or f"sq_bogie12_{args.model}_seed{args.seed}"
        existing = results_dir() / f"{stem}.csv"
        merge_task_results(existing, summary_rows, batch_rows, args)
    else:
        write_outputs(summary_rows, batch_rows, args)


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
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
            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_samples += batch_size
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_one_task(task_id: str, args: argparse.Namespace, device: torch.device) -> Path:
    spec = SQ_TASKS[task_id]
    sources = [int(s) for s in spec["sources"]]  # type: ignore[union-attr]
    ckpt_path = Path(checkpoint_for_task(args.model, task_id))
    if ckpt_path.is_file() and not args.force_train:
        print(f"[TRAIN] skip {task_id} ({sources_tag(sources)}), exists: {ckpt_path}")
        return ckpt_path

    train_speeds = tuple(sources)
    val_speeds = train_speeds
    x_train, y_train, x_val, y_val = get_sq_data(
        train_ratio=args.train_ratio,
        train_speeds=train_speeds,
        test_speeds=val_speeds,
    )
    train_dataset = SQDataset(x_train, y_train, transform=False, augment=False, in_channels=1)
    val_dataset = SQDataset(x_val, y_val, transform=False, augment=False, in_channels=1)
    print(
        f"[TRAIN] {task_id} sources={sources_tag(sources)} "
        f"train={len(train_dataset)} val={len(val_dataset)}"
    )

    num_classes = int(np.max(y_train)) + 1
    cfg = ModelConfig(
        input_length=2048,
        num_classes=num_classes,
        transform_in_model=True,
        zero_mean=True,
        in_channels=1,
        checkpoint_path=None,
    )
    model = build_model(cfg, model_name=args.model, device=device, track_running_stats=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    best_state = None
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_train_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_train_epoch(model, val_loader, criterion, device)
        print(
            f"[TRAIN] {task_id} epoch={epoch:03d} train_acc={train_acc:.2%} val_acc={val_acc:.2%}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "model": _clone_state_dict(model),
                "epoch": epoch,
                "val_acc": val_acc,
                "task_id": task_id,
                "source_speed_idxs": sources,
                "num_classes": num_classes,
                "model_name": args.model,
                "input_length": 2048,
            }

    if best_state is None:
        raise RuntimeError(f"Training failed for {task_id}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)
    print(f"[TRAIN] saved {ckpt_path} val_acc={best_acc:.2%}")
    return ckpt_path


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    tasks = resolve_tasks(args)
    unique_sources: Dict[str, str] = {}
    for task_id in tasks:
        spec = SQ_TASKS[task_id]
        tag = sources_tag(spec["sources"])  # type: ignore[arg-type]
        if tag not in unique_sources:
            unique_sources[tag] = task_id
    for task_id in unique_sources.values():
        train_one_task(task_id, args, device)


def remap_legacy_csv_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        mapped = dict(row)
        scen = str(row.get("scenario", ""))
        task = LEGACY_SCENARIO_TO_TASK.get(scen, scen)
        mapped["task"] = task
        mapped["scenario"] = task
        out.append(mapped)
    return out


def merge_task_results(
    existing_csv: Path,
    new_rows: List[Dict[str, object]],
    batch_rows: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    merged: Dict[Tuple[str, str], Dict[str, object]] = {}
    if existing_csv.is_file():
        with existing_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                remapped = remap_legacy_csv_rows([row])[0]
                key = (str(remapped["task"]), str(remapped["method"]))
                merged[key] = remapped
    for row in new_rows:
        key = (str(row["task"]), str(row["method"]))
        merged[key] = row
    summary_rows = [merged[k] for k in sorted(merged.keys())]
    write_outputs(summary_rows, batch_rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQ eight-task TTA comparison (Bogie 12 methods)")
    parser.add_argument("command", nargs="?", default="eval", choices=["eval", "train", "merge"])
    parser.add_argument("--tasks", type=str, default="", help="T1–T8 comma list or 'all'")
    parser.add_argument("--model", type=str, default="tfn", choices=["resnet18", "tfn", "tfn_sttf", "wdcnn"])
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--scenarios", type=str, default="domain,noise_dyn,gli", help="Legacy alias for T1,T7,T8")
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--protocol", type=str, default="online-batch", choices=sorted(ROTTA_PROTOCOLS))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_speeds", type=str, default="0,1")
    parser.add_argument("--domain_test_speeds", type=str, default="2,3")
    parser.add_argument("--corruption_speeds", type=str, default="0,1")
    parser.add_argument("--gli_gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_stem", type=str, default="")
    parser.add_argument("--method_config", type=str, default="")
    parser.add_argument("--tribe_config", type=str, default="")

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
    parser.add_argument("--tribe_lr", type=float, default=None)
    parser.add_argument("--tribe_optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--tribe_weight_decay", type=float, default=None)
    parser.add_argument("--tribe_steps", type=int, default=None)
    parser.add_argument("--tribe_eta", type=float, default=None)
    parser.add_argument("--tribe_gamma", type=float, default=None)
    parser.add_argument("--tribe_h0", type=float, default=None)
    parser.add_argument("--tribe_lambda", type=float, default=None)
    parser.add_argument("--tribe_gaussian_std", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument(
        "--merge_with_existing",
        action="store_true",
        help="Merge eval results into existing sq_bogie12_* CSV (remaps legacy T1/T7/T8 keys)",
    )
    return parser.parse_args()


def load_method_config(args: argparse.Namespace) -> None:
    args.tribe_config_data = {}
    if args.tribe_config:
        tribe_path = Path(args.tribe_config)
        if not tribe_path.is_file():
            tribe_path = Path(__file__).resolve().parent / "configs" / args.tribe_config
        if not tribe_path.is_file():
            tribe_path = TTA_DIR / "bogie" / "configs" / args.tribe_config
        if tribe_path.is_file():
            args.tribe_config_data = load_tribe_config(tribe_path)

    config_path = resolve_method_config_path(args)
    args.method_config_path = config_path
    args.method_config_data = load_method_config_json(config_path) if config_path.is_file() else {}


def main() -> None:
    args = parse_args()
    load_method_config(args)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"[INFO] device={device}, model={args.model}, command={args.command}")

    if args.command == "train":
        run_train(args, device)
        return

    if args.command == "merge":
        stem = getattr(args, "output_stem", None) or f"sq_bogie12_{args.model}_seed{args.seed}"
        existing = results_dir() / f"{stem}.csv"
        partial = results_dir() / f"{stem}_partial.csv"
        if not partial.is_file():
            raise FileNotFoundError(f"Missing partial results: {partial}")
        with partial.open(newline="", encoding="utf-8") as handle:
            new_rows = list(csv.DictReader(handle))
        merge_task_results(existing, new_rows, [], args)
        return

    run_eval(args, device)


if __name__ == "__main__":
    main()
