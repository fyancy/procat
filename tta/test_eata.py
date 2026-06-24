import argparse
import math
import sys
from copy import deepcopy
from typing import Dict, Optional, Tuple

import warnings

warnings.filterwarnings("ignore")

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from .common import (
        DataConfig,
        LoaderConfig,
        ModelConfig,
        create_sq_dataloaders,
        build_model,
        TTABase,
        softmax_entropy,
        get_default_device,
        parse_speeds_arg,
        evaluate_classification,
    )
except ImportError:
    from common import (
        DataConfig,
        LoaderConfig,
        ModelConfig,
        create_sq_dataloaders,
        build_model,
        TTABase,
        softmax_entropy,
        get_default_device,
        parse_speeds_arg,
        evaluate_classification,
    )


def _append_ndjson(payload: dict):
    # 已移除调试插桩：保留函数名避免大范围改动（兼容历史调用点）
    return


def collect_bn_affine_params(model: nn.Module):
    """
    对齐官方：收集 BN 的仿射参数 weight/bias（WDCNN 是 BatchNorm1d）。
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np}" if nm else np)
    return params, names


def configure_model_for_eata(model: nn.Module) -> nn.Module:
    """
    参考官方 EATA 的 configure_model()：
    - train()（因为要做熵最小化更新）
    - 全部 requires_grad_(False)
    - 仅 BN 开启梯度
    - BN 强制使用 batch stats：track_running_stats=False 且 running_mean/var=None
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.momentum = 0.01
    return model


def update_model_probs(current_model_probs: Optional[torch.Tensor], new_probs: torch.Tensor) -> Optional[torch.Tensor]:
    """
    对齐官方：moving average 概率向量（Eqn.4），动量系数固定 0.9。
    """
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        with torch.no_grad():
            return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            return current_model_probs
        with torch.no_grad():
            return 0.9 * current_model_probs + (1.0 - 0.9) * new_probs.mean(0)


@torch.enable_grad()
def forward_and_adapt_eata(
    x: torch.Tensor,
    model: nn.Module,
    optimizer: optim.Optimizer,
    fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    e_margin: float,
    current_model_probs: Optional[torch.Tensor],
    fisher_alpha: float = 2000.0,
    d_margin: float = 0.05,
):
    """
    参考官方 EATA forward_and_adapt_eata()：
    - 先按 entropy < e_margin 过滤 unreliable
    - 再按与 moving-average probs 的 cosine similarity 过滤 redundant
    - 对选中样本做 reweight entropy loss + 可选 fisher(EWC) 正则
    """
    outputs = model(x)
    entropys = softmax_entropy(outputs)

    # 1) filter unreliable
    filter_ids_1 = torch.where(entropys < e_margin)  # tuple
    selected_1 = filter_ids_1[0]
    ent_1 = entropys[filter_ids_1]

    # 2) filter redundant
    selected_2 = selected_1
    ent_2 = ent_1
    updated_probs = None

    if current_model_probs is not None:
        probs_1 = outputs[filter_ids_1].softmax(1)
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(0), probs_1, dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        selected_2 = selected_1[filter_ids_2[0]]
        ent_2 = ent_1[filter_ids_2]
        updated_probs = update_model_probs(current_model_probs, probs_1[filter_ids_2])
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))

    did_step = False
    loss_val = 0.0
    num_1 = int(selected_1.numel())
    num_2 = int(selected_2.numel())

    if num_2 > 0:
        coeff = 1.0 / torch.exp(ent_2.clone().detach() - e_margin)
        ent_2 = ent_2.mul(coeff)
        loss = ent_2.mean(0)

        if fishers is not None:
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in fishers:
                    fisher_diag, p0 = fishers[name]
                    ewc_loss = ewc_loss + fisher_alpha * (fisher_diag * (param - p0) ** 2).sum()
            loss = loss + ewc_loss

        loss.backward()
        optimizer.step()
        did_step = True
        loss_val = float(loss.detach().item())

    optimizer.zero_grad(set_to_none=True)
    return outputs, num_2, num_1, updated_probs, did_step, loss_val


def compute_fishers_bn_affine(
    model: nn.Module,
    data_loader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    为 EATA 的 EWC 正则计算 fisher（仅 BN 仿射参数）。
    - 使用 source(train) 数据的监督 CE loss 估计 fisher diag
    - 返回：{param_name: (fisher_diag, param_snapshot)}
    """
    model.eval()
    params, names = collect_bn_affine_params(model)
    name_set = set(names)
    fishers: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for n, p in model.named_parameters():
        if n in name_set:
            fishers[n] = (torch.zeros_like(p, device=device), p.detach().clone())

    criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)

    for i, (inputs, targets) in enumerate(data_loader):
        if i >= max_batches:
            break
        inputs = torch.from_numpy(inputs) if hasattr(inputs, "dtype") and not torch.is_tensor(inputs) else inputs
        targets = torch.from_numpy(targets) if hasattr(targets, "dtype") and not torch.is_tensor(targets) else targets
        inputs = inputs.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).long()

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in fishers and p.grad is not None:
                    fishers[n] = (fishers[n][0] + (p.grad.detach() ** 2), fishers[n][1])

        model.zero_grad(set_to_none=True)

    # 平均
    denom = float(max(1, min(max_batches, i + 1)))
    with torch.no_grad():
        for n in list(fishers.keys()):
            fishers[n] = (fishers[n][0] / denom, fishers[n][1])

    return fishers


class EATAOfficial(TTABase):
    """
    EATA 官方逻辑对齐（参考 ICML 2022 EATA repo）：
    - 熵最小化（类似 TENT）
    - 两级过滤：unreliable + redundant
    - moving-average probs
    - 可选 fisher(EWC) 抗遗忘
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        fisher_alpha: float = 2000.0,
        steps: int = 1,
        episodic: bool = False,
        e_margin: float = 0.0,
        d_margin: float = 0.05,
    ):
        super().__init__(model, device=device)
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha
        self.e_margin = e_margin
        self.d_margin = d_margin
        self.current_model_probs: Optional[torch.Tensor] = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

        self._batch_idx = 0

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.current_model_probs = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0

    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()

        self.model.train()
        # self.model(inputs)
        outputs = None
        for _ in range(self.steps):
            (
                outputs,
                num2,
                num1,
                updated_probs,
                did_step,
                loss_val,
            ) = forward_and_adapt_eata(
                inputs,
                self.model,
                self.optimizer,
                self.fishers,
                self.e_margin,
                self.current_model_probs,
                fisher_alpha=self.fisher_alpha,
                d_margin=self.d_margin,
            )
            self.num_samples_update_2 += int(num2)
            self.num_samples_update_1 += int(num1)
            self.current_model_probs = updated_probs

        self._batch_idx += 1
        return outputs.detach() if outputs is not None else self.model(inputs).detach()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQ 数据集 EATA (Official-aligned)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument(
        "--model",
        type=str,
        default="wdcnn",
        choices=["wdcnn", "resnet18", "resnet34", "resnet50", "resnet101"],
        help="选择 backbone 模型",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="",
        help="预训练模型 checkpoint 路径（为空则使用 checkpoints/{model}_sq.pth）",
    )
    p.add_argument(
        "--test_speeds",
        type=str,
        default="0,1",
        help="目标域测试工况 speeds，例如 '0,1' 或 '2,3' 或 '0,1,2,3'",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--episodic", action="store_true")

    # 对齐官方默认：e_margin = log(1000)/2 - 1
    # 参考：https://raw.githubusercontent.com/mr-eggplant/EATA/main/eata.py
    p.add_argument(
        "--e_margin",
        type=float,
        default=None,
        help="Entropy filter threshold. Defaults to 0.5 * log(num_classes).",
    )
    p.add_argument("--d_margin", type=float, default=0.05)

    p.add_argument("--use_fisher", action="store_true")
    p.add_argument("--fisher_alpha", type=float, default=2000.0)
    p.add_argument("--fisher_batches", type=int, default=10)

    p.add_argument("--no_transform", action="store_true")
    p.add_argument("--device", type=str, default="")
    p.add_argument(
        "--corruption_type", 
        type=str, 
        default=None, 
        choices=["noise", "missing"]
        )
    p.add_argument(
        "--severity", 
        type=int, 
        default=5, 
        choices=[0, 1, 2, 3, 4, 5]
        )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else get_default_device()
    print(f"[INFO] Device: {device}")

    data_cfg = DataConfig(
        train_ratio=args.train_ratio,
        cross_domain=False,
        transform=not args.no_transform,
        augment_train=False,
        in_channels=1,
        train_speeds=(0, 1),
        test_speeds=parse_speeds_arg(args.test_speeds),
        corruption_type=args.corruption_type,
        severity=args.severity,
    )
    loader_cfg = LoaderConfig(batch_size=args.batch_size)
    loaders, num_classes = create_sq_dataloaders(data_cfg, loader_cfg)

    model_cfg = ModelConfig(
        input_length=2048,
        num_classes=num_classes,
        transform_in_model=not args.no_transform,
        zero_mean=True,
        in_channels=1,
        checkpoint_path=args.model_path.strip() or f"checkpoints/{args.model}_sq.pth",
    )

    # Baseline (source)
    print("\n=== Baseline (No Adapt) ===")
    base_model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    base_metrics = evaluate_classification(base_model, loaders["test"], device=device, criterion=nn.CrossEntropyLoss())
    print(f"Baseline Acc: {base_metrics['acc']:.4%}")

    # EATA model (adapt)
    model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    model = configure_model_for_eata(model)
    bn_params, bn_names = collect_bn_affine_params(model)
    # bn_params = model.parameters()
    if not bn_params:
        raise RuntimeError("EATA: 未找到任何 BN 仿射参数（weight/bias）。")
    optimizer = optim.Adam(bn_params, lr=args.lr, betas=(0.9, 0.999))

    e_margin = float(args.e_margin) if args.e_margin is not None else 0.5 * math.log(num_classes)

    fishers = None
    if args.use_fisher:
        # fisher 基于 source(train) 计算：用 base_model（不改变 BN 统计）
        fishers = compute_fishers_bn_affine(
            model=base_model,
            data_loader=loaders["train"],
            device=device,
            max_batches=args.fisher_batches,
        )

    print(
        f"\n=== EATA Official-aligned (lr={args.lr}, steps={args.steps}, e_margin={e_margin:.4f}, d_margin={args.d_margin}) ==="
    )
    eata = EATAOfficial(
        model=model,
        optimizer=optimizer,
        device=device,
        fishers=fishers,
        fisher_alpha=args.fisher_alpha,
        steps=args.steps,
        episodic=args.episodic,
        e_margin=e_margin,
        d_margin=args.d_margin,
    )
    metrics = eata.predict_loader(loaders["test"])
    print(f"EATA Acc: {metrics.get('acc', 0.0):.4%}")


if __name__ == "__main__":
    main()


