"""Faithful TRIBE port for 1D/2D models (Gorilla-Lab-SCUT/TRIBE).

Differs from ``tta.tribe.test_tribe.TRIBE`` (local port):
- Student strong aug follows official ``get_tta_transforms`` (not RoTTA ``make_rotta_strong_view``).
- Balanced BN uses official ``update_statistics_*_v5`` (not the local scatter reimplementation).
- Source-anchor MSE masks element-wise logits then averages (official reduction).
- Default hyperparameters match ``tribe_config_official.json`` (lambda=0.5, eta=0.01, ...).
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    from ..common import TTABase
    from ..test_rotta import ROTTA_PROTOCOLS, _gaussian_blur_1d, _time_shift
    from .balanced_bn_official import BalancedRobustBN1dOfficial, BalancedRobustBN2dOfficial
except ImportError:
    from common import TTABase
    from test_rotta import ROTTA_PROTOCOLS, _gaussian_blur_1d, _time_shift
    from balanced_bn_official import BalancedRobustBN1dOfficial, BalancedRobustBN2dOfficial


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)
OFFICIAL_BN_TYPES = (BalancedRobustBN1dOfficial, BalancedRobustBN2dOfficial)
TRIBE_RUNTIME_PROTOCOLS = {"standalone", "online", "online-batch", "online-batch-bias"}

OFFICIAL_DEFAULT_LR = 1e-3
OFFICIAL_DEFAULT_STEPS = 1
OFFICIAL_DEFAULT_ETA = 0.01
OFFICIAL_DEFAULT_GAMMA = 0.0
OFFICIAL_DEFAULT_LAMBDA = 0.5
OFFICIAL_DEFAULT_H0 = 0.05
OFFICIAL_DEFAULT_GAUSSIAN_STD = 0.005
OFFICIAL_DEFAULT_ONLINE_BATCH_SIZE = 64

CONFIG_PATH = Path(__file__).resolve().parents[1] / "bogie" / "configs" / "tribe_config_official.json"


def load_official_config(path: Optional[Path] = None) -> Dict[str, object]:
    cfg_path = path or CONFIG_PATH
    if not cfg_path.is_file():
        return {}
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return dict(data.get("tribe", data))


def get_official_defaults() -> Dict[str, object]:
    base = {
        "lr": OFFICIAL_DEFAULT_LR,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "steps": OFFICIAL_DEFAULT_STEPS,
        "eta": OFFICIAL_DEFAULT_ETA,
        "gamma": OFFICIAL_DEFAULT_GAMMA,
        "lambda_reg": OFFICIAL_DEFAULT_LAMBDA,
        "h0": OFFICIAL_DEFAULT_H0,
        "gaussian_std": OFFICIAL_DEFAULT_GAUSSIAN_STD,
        "online_batch_size": OFFICIAL_DEFAULT_ONLINE_BATCH_SIZE,
    }
    base.update(load_official_config())
    return base


def get_named_submodule(model: nn.Module, name: str) -> nn.Module:
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


def tribe_official_strong_view_1d(
    x: torch.Tensor,
    gaussian_std: float = OFFICIAL_DEFAULT_GAUSSIAN_STD,
    soft: bool = False,
) -> torch.Tensor:
    """1D analogue of official ``get_tta_transforms`` for Conv1d inputs [B, C, L]."""
    out = x
    brightness = (0.8, 1.2) if soft else (0.6, 1.4)
    contrast_range = (0.85, 1.15) if soft else (0.7, 1.3)
    scale_range = (0.95, 1.05) if soft else (0.9, 1.1)
    blur_sigma = (0.001, 0.25) if soft else (0.001, 0.5)

    batch = out.size(0)
    device = out.device
    mean = out.mean(dim=-1, keepdim=True)
    contrast = torch.empty(batch, 1, 1, device=device).uniform_(*contrast_range)
    gain = torch.empty(batch, 1, 1, device=device).uniform_(*brightness)
    out = (out - mean) * contrast + mean
    out = out * gain
    out = _time_shift(out, 1.0 / 16.0)
    scale = torch.empty(batch, 1, 1, device=device).uniform_(*scale_range)
    out = out * scale
    sigma = float(torch.empty(1, device=device).uniform_(*blur_sigma).item())
    out = _gaussian_blur_1d(out, sigma=sigma)
    if torch.rand(1, device=device).item() < 0.5:
        out = out.flip(dims=[-1])
    out = out + torch.randn_like(out) * gaussian_std
    return out.clamp(-1.0, 1.0)


def configure_tribe_official_model(
    model: nn.Module,
    num_classes: int,
    eta: float,
    gamma: float,
) -> List[nn.Parameter]:
    model.requires_grad_(False)
    norm_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, BN_TYPES):
            norm_names.append(name)

    for name in norm_names:
        bn_layer = get_named_submodule(model, name)
        if isinstance(bn_layer, nn.BatchNorm1d):
            new_bn = BalancedRobustBN1dOfficial(bn_layer, num_classes=num_classes, momentum=eta, gamma=gamma)
        elif isinstance(bn_layer, nn.BatchNorm2d):
            new_bn = BalancedRobustBN2dOfficial(bn_layer, num_classes=num_classes, momentum=eta, gamma=gamma)
        else:
            raise RuntimeError(f"Unsupported BN layer: {name}")
        new_bn.requires_grad_(True)
        set_named_submodule(model, name, new_bn)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("TRIBE official selected no trainable parameters.")
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


class TRIBE_OFFICIAL(TTABase):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        lr: float = OFFICIAL_DEFAULT_LR,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        steps: int = OFFICIAL_DEFAULT_STEPS,
        protocol: Optional[str] = None,
        eta: float = OFFICIAL_DEFAULT_ETA,
        gamma: float = OFFICIAL_DEFAULT_GAMMA,
        lambda_reg: float = OFFICIAL_DEFAULT_LAMBDA,
        h0: float = OFFICIAL_DEFAULT_H0,
        gaussian_std: float = OFFICIAL_DEFAULT_GAUSSIAN_STD,
        online_batch_size: int = OFFICIAL_DEFAULT_ONLINE_BATCH_SIZE,
    ) -> None:
        adapt_mode = "batch"
        reset_each_sample = False
        if protocol:
            protocol = protocol.lower()
            if protocol not in TRIBE_RUNTIME_PROTOCOLS:
                raise ValueError(f"protocol must be one of: {sorted(TRIBE_RUNTIME_PROTOCOLS)}")
            adapt_mode = "batch" if protocol in {"online-batch", "online-batch-bias"} else "sample"
            reset_each_sample = protocol == "standalone"

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
        self.gaussian_std = float(gaussian_std)
        self.adapt_mode = adapt_mode
        self.reset_each_sample = bool(reset_each_sample)
        self.online_batch_size = max(int(online_batch_size), 1)
        self.last_loss: Optional[float] = None
        self.last_confident_count: int = 0

        self.params = configure_tribe_official_model(
            self.model, num_classes=self.num_classes, eta=self.eta, gamma=self.gamma
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
            if isinstance(module, OFFICIAL_BN_TYPES):
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

        strong_sup_aug = tribe_official_strong_view_1d(batch_data, gaussian_std=self.gaussian_std)

        self.set_bn_label(self.aux_model, pseudo_label)
        ema_sup_out = self.aux_model(batch_data)

        self.set_bn_label(self.model, pseudo_label)
        stu_sup_out = self.model(strong_sup_aug)

        entropy = self.self_softmax_entropy(ema_sup_out)
        entropy_mask = entropy < self.h0 * math.log(max(self.num_classes, 2))
        self.last_confident_count = int(entropy_mask.sum().item())
        if self.last_confident_count == 0:
            self.last_loss = None
            return

        l_sup = F.cross_entropy(stu_sup_out, ema_sup_out.argmax(dim=-1), reduction="none")[entropy_mask].mean()

        with torch.no_grad():
            self.set_bn_label(self.source_model, pseudo_label)
            source_anchor = self.source_model(batch_data).detach()

        l_reg = self.lambda_reg * F.mse_loss(ema_sup_out, source_anchor, reduction="none")[entropy_mask].mean()
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
