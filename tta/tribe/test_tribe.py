"""TRIBE test-time adaptation for SQ 1D signals.

This is a local port of Gorilla-Lab-SCUT/TRIBE:

- Balanced BatchNorm with class-conditional running statistics;
- tri-net self-training with a student model, an auxiliary teacher branch that
  shares trainable affine parameters with the student, and a source-anchor
  branch;
- low-entropy pseudo-label filtering and source-anchor regularization.

The original repository targets image tensors and has a 1D toy layer using a
``[B, L, C]`` convention. The SQ models here use PyTorch Conv1d tensors
``[B, C, L]``, so the balanced-statistics code below keeps the TRIBE update
formula but adapts the tensor dimensions to the local networks.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

THIS_DIR = Path(__file__).resolve().parent
TTA_DIR = THIS_DIR.parent
ROOT_DIR = TTA_DIR.parent
for path in (str(ROOT_DIR), str(TTA_DIR), str(THIS_DIR)):
    if path not in sys.path:
        sys.path.append(path)

try:
    from ..common import (
        DataConfig,
        LoaderConfig,
        ModelConfig,
        TTABase,
        build_model,
        create_sq_dataloaders,
        evaluate_classification,
        get_default_device,
        parse_speeds_arg,
    )
    from ..test_rotta import (
        ROTTA_PROTOCOLS,
        make_rotta_strong_view,
    )
except ImportError:
    from common import (
        DataConfig,
        LoaderConfig,
        ModelConfig,
        TTABase,
        build_model,
        create_sq_dataloaders,
        evaluate_classification,
        get_default_device,
        parse_speeds_arg,
    )
    from test_rotta import (
        ROTTA_PROTOCOLS,
        make_rotta_strong_view,
    )


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)
TRIBE_PROTOCOLS = ROTTA_PROTOCOLS
TRIBE_RUNTIME_PROTOCOLS = {"standalone", "online", "online-batch", "online-batch-bias"}

# Robust default selected from online-batch tuning on domain/noise scenarios.
DEFAULT_TRIBE_LR = 3e-4
DEFAULT_TRIBE_STEPS = 1
DEFAULT_TRIBE_ETA = 0.1
DEFAULT_TRIBE_GAMMA = 0.1
DEFAULT_TRIBE_LAMBDA = 0.0
DEFAULT_TRIBE_H0 = 0.05
DEFAULT_TRIBE_NOISE_STD = 0.0
DEFAULT_TRIBE_ONLINE_BATCH_SIZE = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_named_submodule(model: nn.Module, name: str) -> nn.Module:
    if not name:
        return model
    current = model
    for part in name.split("."):
        current = getattr(current, part)
    return current


def set_named_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = get_named_submodule(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], module)


def set_named_parameter(model: nn.Module, name: str, parameter: nn.Parameter) -> None:
    parts = name.split(".")
    parent = get_named_submodule(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], parameter)


def _copy_affine_parameter(parameter: Optional[nn.Parameter]) -> Optional[nn.Parameter]:
    if parameter is None:
        return None
    return nn.Parameter(parameter.detach().clone())


class BalancedBatchNormBase(nn.Module):
    """Balanced BN statistics used by TRIBE.

    ``global_mean``/``global_var`` are used for normalization. During training,
    pseudo labels update the per-class local statistics first, then aggregate
    them back into the global statistics.
    """

    def __init__(
        self,
        bn_layer: nn.modules.batchnorm._BatchNorm,
        num_classes: int,
        momentum: float = DEFAULT_TRIBE_ETA,
        gamma: float = DEFAULT_TRIBE_GAMMA,
    ) -> None:
        super().__init__()
        if bn_layer.running_mean is None or bn_layer.running_var is None:
            raise ValueError("TRIBE BalancedBN requires source BatchNorm running statistics.")

        self.num_features = int(bn_layer.num_features)
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.gamma = float(gamma)
        self.eps = float(bn_layer.eps)

        running_mean = bn_layer.running_mean.detach().clone()
        running_var = bn_layer.running_var.detach().clone()
        self.register_buffer("global_mean", running_mean)
        self.register_buffer("global_var", running_var)
        self.register_buffer("local_mean", running_mean.unsqueeze(0).repeat(self.num_classes, 1).clone())
        self.register_buffer("local_var", running_var.unsqueeze(0).repeat(self.num_classes, 1).clone())

        weight = _copy_affine_parameter(bn_layer.weight)
        bias = _copy_affine_parameter(bn_layer.bias)
        if weight is None:
            self.register_parameter("weight", None)
        else:
            self.weight = weight
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = bias

        self.label: Optional[torch.Tensor] = None

    def _sample_feature_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            return x, x.pow(2)
        reduce_dims = tuple(range(2, x.dim()))
        return x.mean(dim=reduce_dims), x.pow(2).mean(dim=reduce_dims)

    def _view_shape(self, x: torch.Tensor) -> Tuple[int, ...]:
        return (1, int(x.shape[1]), *([1] * (x.dim() - 2)))

    def _batch_stats_dims(self, x: torch.Tensor) -> List[int]:
        return [0] + list(range(2, x.dim()))

    @torch.no_grad()
    def _update_without_labels(self, x: torch.Tensor) -> None:
        b_var, b_mean = torch.var_mean(x, dim=self._batch_stats_dims(x), unbiased=False, keepdim=False)
        self.global_mean.mul_(1.0 - self.momentum).add_(b_mean, alpha=self.momentum)
        self.global_var.mul_(1.0 - self.momentum).add_(b_var, alpha=self.momentum)
        self.global_var.clamp_(min=1e-6)

    @torch.no_grad()
    def _update_with_labels(self, x: torch.Tensor, label: torch.Tensor) -> None:
        if x.size(0) == 0:
            return
        label = label.detach().to(device=x.device, dtype=torch.long).reshape(-1)
        if label.numel() != x.size(0):
            raise RuntimeError(f"TRIBE label/data length mismatch: {label.numel()} vs {x.size(0)}")
        label = label.clamp_(0, self.num_classes - 1)

        unique_labels = label.unique(sorted=True)
        if unique_labels.numel() == 0:
            return

        reverse = label.new_full((self.num_classes,), -1)
        reverse[unique_labels] = torch.arange(unique_labels.numel(), device=label.device, dtype=label.dtype)
        label_local = reverse[label]

        sample_mean, sample_square = self._sample_feature_stats(x)
        local_mean_for_sample = self.local_mean[label]
        delta_mean_sample = sample_mean - local_mean_for_sample

        counts = sample_mean.new_zeros((unique_labels.numel(),))
        counts.scatter_add_(0, label_local, torch.ones_like(label_local, dtype=sample_mean.dtype))
        per_class_momentum = torch.where(
            counts > (1.0 / max(self.momentum, 1e-12)),
            counts.reciprocal(),
            torch.full_like(counts, self.momentum),
        )

        delta_k = sample_mean.new_zeros((unique_labels.numel(), self.num_features))
        delta_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, self.num_features), delta_mean_sample)
        delta_k.mul_(per_class_momentum.unsqueeze(-1))

        old_local_mean = self.local_mean[unique_labels].clone()
        old_local_var = self.local_var[unique_labels].clone()
        self.local_mean[unique_labels] = old_local_mean + (1.0 - self.gamma) * delta_k
        self.local_mean.add_(self.gamma * delta_k.mean(dim=0, keepdim=True))

        delta_square_sample = sample_square - 2.0 * local_mean_for_sample * sample_mean + local_mean_for_sample.pow(2)
        delta_square_k = sample_mean.new_zeros((unique_labels.numel(), self.num_features))
        delta_square_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, self.num_features), delta_square_sample)

        var_delta = (
            per_class_momentum.unsqueeze(-1)
            * (delta_square_k - counts.unsqueeze(-1) * old_local_var)
            - delta_k.pow(2)
        )
        self.local_var[unique_labels] = old_local_var + (1.0 - self.gamma) * var_delta

        var_gap = (per_class_momentum.unsqueeze(-1) * delta_square_k - delta_k.pow(2)).mean(dim=0, keepdim=True)
        mean_m_count = (per_class_momentum * counts).mean()
        self.local_var.add_(self.gamma * (var_gap - mean_m_count * self.local_var))
        self.local_var.clamp_(min=1e-6)

        self.global_mean.copy_(self.local_mean.mean(dim=0))
        self.global_var.copy_(self.local_var.mean(dim=0) + self.local_mean.var(dim=0, unbiased=False))
        self.global_var.clamp_(min=1e-6)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        shape = self._view_shape(x)
        out = (x - self.global_mean.view(shape)) / torch.sqrt(self.global_var.view(shape) + self.eps)
        if self.weight is not None:
            out = out * self.weight.view(shape)
        if self.bias is not None:
            out = out + self.bias.view(shape)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.label is not None:
                self._update_with_labels(x, self.label)
                self.label = None
            else:
                self._update_without_labels(x)
        return self._normalize(x)


class BalancedRobustBN1d(BalancedBatchNormBase):
    pass


class BalancedRobustBN2d(BalancedBatchNormBase):
    pass


def configure_tribe_model(
    model: nn.Module,
    num_classes: int,
    eta: float,
    gamma: float,
    adapt_params: str = "affine",
) -> List[nn.Parameter]:
    if adapt_params not in {"affine", "bias"}:
        raise ValueError("TRIBE adapt_params must be 'affine' or 'bias'.")

    model.requires_grad_(False)
    normlayer_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, BN_TYPES):
            normlayer_names.append(name)

    for name in normlayer_names:
        bn_layer = get_named_submodule(model, name)
        if isinstance(bn_layer, nn.BatchNorm1d):
            new_bn = BalancedRobustBN1d(bn_layer, num_classes=num_classes, momentum=eta, gamma=gamma)
        elif isinstance(bn_layer, nn.BatchNorm2d):
            new_bn = BalancedRobustBN2d(bn_layer, num_classes=num_classes, momentum=eta, gamma=gamma)
        else:
            raise RuntimeError(f"Unsupported BN layer: {name}")

        new_bn.requires_grad_(False)
        if new_bn.weight is not None:
            new_bn.weight.requires_grad = adapt_params == "affine"
        if new_bn.bias is not None:
            new_bn.bias.requires_grad = True
        set_named_submodule(model, name, new_bn)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("TRIBE selected no trainable parameters.")
    return params


def make_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    if optimizer_name == "adam":
        return optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown TRIBE optimizer: {optimizer_name}")


def _share_parameters(target_model: nn.Module, source_model: nn.Module) -> None:
    source_params = dict(source_model.named_parameters())
    for name in list(dict(target_model.named_parameters()).keys()):
        set_named_parameter(target_model, name, source_params[name])


class TRIBE(TTABase):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        lr: float = DEFAULT_TRIBE_LR,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        steps: int = DEFAULT_TRIBE_STEPS,
        protocol: Optional[str] = None,
        eta: float = DEFAULT_TRIBE_ETA,
        gamma: float = DEFAULT_TRIBE_GAMMA,
        lambda_reg: float = DEFAULT_TRIBE_LAMBDA,
        h0: float = DEFAULT_TRIBE_H0,
        adapt_params: str = "affine",
        reset_each_sample: bool = False,
        adapt_mode: str = "sample",
        online_batch_size: int = DEFAULT_TRIBE_ONLINE_BATCH_SIZE,
        noise_std: float = DEFAULT_TRIBE_NOISE_STD,
    ) -> None:
        if protocol:
            protocol = protocol.lower()
            if protocol not in TRIBE_RUNTIME_PROTOCOLS:
                raise ValueError(f"protocol must be one of: {sorted(TRIBE_RUNTIME_PROTOCOLS)}")
            adapt_mode = "batch" if protocol in {"online-batch", "online-batch-bias"} else "sample"
            reset_each_sample = protocol == "standalone"
            if protocol == "online-batch-bias":
                adapt_params = "bias"
        if adapt_mode not in {"sample", "batch"}:
            raise ValueError("adapt_mode must be 'sample' or 'batch'.")

        super().__init__(model, device=device)
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self.optimizer_name = optimizer_name
        self.weight_decay = float(weight_decay)
        self.steps = int(steps)
        self.protocol = protocol
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.lambda_reg = float(lambda_reg)
        self.h0 = float(h0)
        self.adapt_params = adapt_params
        self.reset_each_sample = bool(reset_each_sample)
        self.adapt_mode = adapt_mode
        self.online_batch_size = max(int(online_batch_size), 1)
        self.noise_std = float(noise_std)
        self.last_loss: Optional[float] = None
        self.last_confident_count: int = 0

        self.params = configure_tribe_model(
            self.model,
            num_classes=self.num_classes,
            eta=self.eta,
            gamma=self.gamma,
            adapt_params=self.adapt_params,
        )
        self.optimizer = make_optimizer(
            self.params,
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.aux_model = copy.deepcopy(self.model).to(self.device)
        _share_parameters(self.aux_model, self.model)
        self.source_model = copy.deepcopy(self.model).to(self.device)
        self.source_model.requires_grad_(False)
        for param in self.aux_model.parameters():
            # These are shared Parameter objects; the student optimizer owns them.
            param.requires_grad = param.requires_grad

        self.initial_model_state = copy.deepcopy(self.model.state_dict())
        self.initial_aux_state = copy.deepcopy(self.aux_model.state_dict())
        self.initial_source_state = copy.deepcopy(self.source_model.state_dict())
        self.initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self.initial_model_state, strict=True)
        self.aux_model.load_state_dict(self.initial_aux_state, strict=True)
        self.source_model.load_state_dict(self.initial_source_state, strict=True)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        self.last_loss = None
        self.last_confident_count = 0

    def reset_for_new_sample(self) -> None:
        self.reset()

    @staticmethod
    def set_bn_label(model: nn.Module, label: Optional[torch.Tensor] = None) -> None:
        for module in model.modules():
            if isinstance(module, (BalancedRobustBN1d, BalancedRobustBN2d)):
                module.label = label

    @staticmethod
    def self_softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
        return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)

    @torch.no_grad()
    def _teacher_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.aux_model.eval()
        return self.aux_model(inputs)

    @torch.no_grad()
    def predict_current(self, inputs: torch.Tensor) -> torch.Tensor:
        self.aux_model.eval()
        return self.aux_model(inputs).detach()

    @torch.enable_grad()
    def update_model(self, batch_data: torch.Tensor, teacher_logits: torch.Tensor) -> None:
        pseudo_label = teacher_logits.argmax(dim=1)

        self.source_model.train()
        self.aux_model.train()
        self.model.train()

        strong_sup_aug = make_rotta_strong_view(batch_data, noise_std=self.noise_std)

        self.set_bn_label(self.aux_model, pseudo_label)
        ema_sup_out = self.aux_model(batch_data)

        self.set_bn_label(self.model, pseudo_label)
        stu_sup_out = self.model(strong_sup_aug)

        entropy = self.self_softmax_entropy(ema_sup_out)
        threshold = self.h0 * math.log(max(self.num_classes, 2))
        entropy_mask = entropy < threshold
        self.last_confident_count = int(entropy_mask.sum().item())
        if self.last_confident_count == 0:
            self.last_loss = None
            return

        pseudo_target = ema_sup_out.argmax(dim=-1)
        l_sup = F.cross_entropy(stu_sup_out, pseudo_target, reduction="none")[entropy_mask].mean()

        with torch.no_grad():
            self.set_bn_label(self.source_model, pseudo_label)
            source_anchor = self.source_model(batch_data).detach()

        per_sample_reg = F.mse_loss(ema_sup_out, source_anchor, reduction="none").mean(dim=1)
        l_reg = self.lambda_reg * per_sample_reg[entropy_mask].mean()
        loss = l_sup + l_reg

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.detach().item())

    def forward_and_adapt(self, batch_data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_logits = self._teacher_predict(batch_data)
        self.update_model(batch_data, teacher_logits.detach())
        return teacher_logits.detach()

    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.adapt_mode == "batch":
            logits_parts: List[torch.Tensor] = []
            for start in range(0, inputs.size(0), self.online_batch_size):
                chunk = inputs[start : start + self.online_batch_size]
                outputs = None
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(chunk)
                logits_parts.append(outputs.detach())
            return torch.cat(logits_parts, dim=0)

        logits_parts = []
        for sample_idx in range(inputs.size(0)):
            if self.reset_each_sample:
                self.reset()
            sample = inputs[sample_idx : sample_idx + 1]
            outputs = None
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(sample)
            logits_parts.append(outputs.detach())
        return torch.cat(logits_parts, dim=0)


def run_tribe_evaluation(
    loaders,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
    num_classes: int,
    protocol: str = "online-batch",
    lr: float = DEFAULT_TRIBE_LR,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    steps: int = DEFAULT_TRIBE_STEPS,
    eta: float = DEFAULT_TRIBE_ETA,
    gamma: float = DEFAULT_TRIBE_GAMMA,
    lambda_reg: float = DEFAULT_TRIBE_LAMBDA,
    h0: float = DEFAULT_TRIBE_H0,
    online_batch_size: int = DEFAULT_TRIBE_ONLINE_BATCH_SIZE,
    noise_std: float = DEFAULT_TRIBE_NOISE_STD,
) -> float:
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    tribe = TRIBE(
        model=model,
        device=device,
        num_classes=num_classes,
        protocol=protocol,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        steps=steps,
        eta=eta,
        gamma=gamma,
        lambda_reg=lambda_reg,
        h0=h0,
        online_batch_size=online_batch_size,
        noise_std=noise_std,
    )
    return float(tribe.predict_loader(loaders["test"]).get("acc", 0.0))


def run_tribe_store_evaluation(
    loaders,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
    num_classes: int,
    lr: float = DEFAULT_TRIBE_LR,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    steps: int = DEFAULT_TRIBE_STEPS,
    eta: float = DEFAULT_TRIBE_ETA,
    gamma: float = DEFAULT_TRIBE_GAMMA,
    lambda_reg: float = DEFAULT_TRIBE_LAMBDA,
    h0: float = DEFAULT_TRIBE_H0,
    online_batch_size: int = DEFAULT_TRIBE_ONLINE_BATCH_SIZE,
    noise_std: float = DEFAULT_TRIBE_NOISE_STD,
) -> float:
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    tribe = TRIBE(
        model=model,
        device=device,
        num_classes=num_classes,
        protocol="online-batch",
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        steps=steps,
        eta=eta,
        gamma=gamma,
        lambda_reg=lambda_reg,
        h0=h0,
        online_batch_size=online_batch_size,
        noise_std=noise_std,
    )

    chunks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    total_correct = 0
    total_samples = 0

    for inputs, labels in loaders["test"]:
        inputs = torch.from_numpy(inputs) if isinstance(inputs, np.ndarray) else inputs
        labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
        chunks.append((inputs.detach().cpu().float(), labels.detach().cpu().long()))

        tribe.reset()
        for chunk_inputs, _ in chunks:
            chunk_inputs = chunk_inputs.to(device, non_blocking=True).float()
            tribe.adapt_one_batch(chunk_inputs)

        current_inputs = inputs.to(device, non_blocking=True).float()
        current_labels = labels.to(device, non_blocking=True).long()
        with torch.no_grad():
            logits = tribe.predict_current(current_inputs)
        preds = logits.argmax(dim=1)
        total_correct += (preds == current_labels).sum().item()
        total_samples += current_labels.numel()

    return total_correct / max(total_samples, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQ TRIBE evaluation")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="checkpoints/resnet18_sq_clean_noaug.pth")
    parser.add_argument("--test_speeds", type=str, default="2,3")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--protocol", type=str, default="online-batch", choices=sorted(TRIBE_PROTOCOLS))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_speeds", type=str, default="0,1")
    parser.add_argument("--corruption_type", type=str, default=None, choices=["noise", "missing"])
    parser.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--dataset_transform", action="store_true")
    parser.add_argument("--no_model_transform", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRIBE_LR)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=DEFAULT_TRIBE_STEPS)
    parser.add_argument("--eta", type=float, default=DEFAULT_TRIBE_ETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_TRIBE_GAMMA)
    parser.add_argument("--lambda_reg", type=float, default=DEFAULT_TRIBE_LAMBDA)
    parser.add_argument("--h0", type=float, default=DEFAULT_TRIBE_H0)
    parser.add_argument("--noise_std", type=float, default=DEFAULT_TRIBE_NOISE_STD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_default_device()
    data_cfg = DataConfig(
        train_ratio=args.train_ratio,
        cross_domain=False,
        transform=args.dataset_transform,
        augment_train=False,
        in_channels=1,
        train_speeds=parse_speeds_arg(args.train_speeds),
        test_speeds=parse_speeds_arg(args.test_speeds),
        corruption_type=args.corruption_type,
        severity=args.severity,
    )
    batch_size = 1 if args.protocol in {"standalone", "online"} else args.batch_size
    loader_cfg = LoaderConfig(batch_size=batch_size, shuffle_test=False)
    loaders, num_classes = create_sq_dataloaders(data_cfg, loader_cfg)
    model_cfg = ModelConfig(
        input_length=2048,
        num_classes=num_classes,
        transform_in_model=not args.no_model_transform,
        zero_mean=True,
        in_channels=1,
        checkpoint_path=args.model_path,
    )

    base_model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    baseline = evaluate_classification(base_model, loaders["test"], device=device, criterion=nn.CrossEntropyLoss())

    common_kwargs = {
        "loaders": loaders,
        "model_cfg": model_cfg,
        "model_name": args.model,
        "device": device,
        "num_classes": num_classes,
        "lr": args.lr,
        "optimizer_name": args.optimizer,
        "weight_decay": args.weight_decay,
        "steps": args.steps,
        "eta": args.eta,
        "gamma": args.gamma,
        "lambda_reg": args.lambda_reg,
        "h0": args.h0,
        "online_batch_size": args.batch_size,
        "noise_std": args.noise_std,
    }
    if args.protocol == "online-batch-store":
        acc = run_tribe_store_evaluation(**common_kwargs)
    else:
        acc = run_tribe_evaluation(protocol=args.protocol, **common_kwargs)
    print(f"Baseline Acc: {float(baseline.get('acc', 0.0)):.4%}")
    print(f"TRIBE Acc: {acc:.4%}")


if __name__ == "__main__":
    main()
