"""RoTTA test-time adaptation for SQ 1D signals.

This ports the official BIT-DA/RoTTA method structure to the local SQ models:

- category-balanced, uncertainty-and-timeliness memory (CSTU);
- RobustBN replacement for BatchNorm layers;
- EMA teacher prediction and pseudo labels;
- periodic student updates from memory using strong target augmentations;
- EMA update from student to teacher.

The image augmentation in the official code is replaced by a 1D signal analogue
with the same role: strong stochastic target perturbation before the student
update.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("..")

try:
    from .common import (
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


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)
ROTTA_PROTOCOLS = {
    "standalone",
    "online",
    "online-batch",
    "online-batch-bias",
    "online-batch-store",
}
ROTTA_RUNTIME_PROTOCOLS = {"standalone", "online", "online-batch", "online-batch-bias"}

DEFAULT_ROTTA_LR = 1e-3
DEFAULT_ROTTA_STEPS = 1
DEFAULT_ROTTA_MEMORY_SIZE = 64
DEFAULT_ROTTA_UPDATE_FREQUENCY = 64
DEFAULT_ROTTA_LAMBDA_T = 0.0
DEFAULT_ROTTA_LAMBDA_U = 1.0
DEFAULT_ROTTA_ALPHA = 0.5
DEFAULT_ROTTA_NU = 0.001
DEFAULT_ROTTA_NOISE_STD = 0.005
DEFAULT_ROTTA_ONLINE_BATCH_SIZE = 64
DEFAULT_ROTTA_AGE_LOSS_WEIGHT = 0.0
DEFAULT_ROTTA_TEACHER_VIEW = "identity"
DEFAULT_ROTTA_WEAK_NOISE_STD = 0.0005
DEFAULT_ROTTA_WEAK_SHIFT_RATIO = 1.0 / 256.0
DEFAULT_ROTTA_WEAK_GAIN_DELTA = 0.01
DEFAULT_ROTTA_WEAK_CONTRAST_DELTA = 0.01


@dataclass
class MemoryItem:
    data: torch.Tensor
    uncertainty: float = 0.0
    age: int = 0

    def increase_age(self) -> None:
        self.age += 1


class CSTU:
    """Class-balanced memory from RoTTA.

    The eviction score follows the official implementation:
    lambda_t * sigmoid(age / capacity) + lambda_u * uncertainty / log(num_class).
    Lower-score new instances replace higher-score old instances.
    """

    def __init__(self, capacity: int, num_class: int, lambda_t: float = 1.0, lambda_u: float = 1.0):
        self.capacity = int(capacity)
        self.num_class = int(num_class)
        self.per_class = self.capacity / self.num_class
        self.lambda_t = float(lambda_t)
        self.lambda_u = float(lambda_u)
        self.data: List[List[MemoryItem]] = [[] for _ in range(self.num_class)]

    def clone(self) -> "CSTU":
        return copy.deepcopy(self)

    def get_occupancy(self) -> int:
        return sum(len(class_list) for class_list in self.data)

    def per_class_dist(self) -> List[int]:
        return [len(class_list) for class_list in self.data]

    def add_instance(self, instance: Tuple[torch.Tensor, int, float]) -> None:
        x, prediction, uncertainty = instance
        prediction = int(prediction)
        if prediction < 0 or prediction >= self.num_class:
            return
        new_item = MemoryItem(data=x.detach(), uncertainty=float(uncertainty), age=0)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls: int, score: float) -> bool:
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            return self.remove_from_classes(self.get_majority_classes(), score)
        return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes: List[int], score_base: float) -> bool:
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(age=item.age, uncertainty=item.uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls
        if max_class is not None and max_index is not None and max_score is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            return False
        return True

    def get_majority_classes(self) -> List[int]:
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        return [cls for cls, occupied in enumerate(per_class_dist) if occupied == max_occupied]

    def heuristic_score(self, age: int, uncertainty: float) -> float:
        return self.lambda_t * 1.0 / (1.0 + math.exp(-age / self.capacity)) + (
            self.lambda_u * float(uncertainty) / math.log(self.num_class)
        )

    def add_age(self) -> None:
        for class_list in self.data:
            for item in class_list:
                item.increase_age()

    def get_memory(self) -> Tuple[List[torch.Tensor], List[float]]:
        memory_data: List[torch.Tensor] = []
        memory_age: List[float] = []
        for class_list in self.data:
            for item in class_list:
                memory_data.append(item.data)
                memory_age.append(item.age / self.capacity)
        return memory_data, memory_age


class GlobalCSTU(CSTU):
    """CSTU memory without the class-balanced quota."""

    def remove_instance(self, cls: int, score: float) -> bool:
        if self.get_occupancy() < self.capacity:
            return True
        return self.remove_from_classes(list(range(self.num_class)), score)


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.modules.batchnorm._BatchNorm, momentum: float):
        super().__init__()
        if not bn_layer.track_running_stats or bn_layer.running_mean is None or bn_layer.running_var is None:
            raise ValueError("RoTTA RobustBN requires source BatchNorm running statistics.")
        self.num_features = bn_layer.num_features
        self.momentum = float(momentum)
        self.register_buffer("source_mean", copy.deepcopy(bn_layer.running_mean.detach()))
        self.register_buffer("source_var", copy.deepcopy(bn_layer.running_var.detach()))
        self.weight = nn.Parameter(copy.deepcopy(bn_layer.weight.detach()))
        self.bias = nn.Parameter(copy.deepcopy(bn_layer.bias.detach()))
        self.eps = float(bn_layer.eps)

    @staticmethod
    def _stats_dims(x: torch.Tensor) -> List[int]:
        return [0] + list(range(2, x.dim()))

    @staticmethod
    def _view_shape(x: torch.Tensor) -> Tuple[int, ...]:
        return (1, int(x.shape[1]), *([1] * (x.dim() - 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=self._stats_dims(x), unbiased=False, keepdim=False)
            mean = (1.0 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1.0 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean = mean.detach().clone()
            self.source_var = var.detach().clone()
        else:
            mean = self.source_mean
            var = self.source_var

        shape = self._view_shape(x)
        x = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return x * self.weight.view(shape) + self.bias.view(shape)


class RobustBN1d(MomentumBN):
    pass


class RobustBN2d(MomentumBN):
    pass


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


def configure_rotta_model(
    model: nn.Module,
    alpha: float,
    adapt_params: str = "affine",
    use_robust_bn: bool = True,
) -> List[nn.Parameter]:
    if adapt_params not in {"affine", "bias"}:
        raise ValueError("RoTTA adapt_params must be 'affine' or 'bias'.")

    model.requires_grad_(False)
    normlayer_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, BN_TYPES):
            normlayer_names.append(name)

    for name in normlayer_names:
        bn_layer = get_named_submodule(model, name)
        if use_robust_bn:
            if isinstance(bn_layer, nn.BatchNorm1d):
                bn_layer = RobustBN1d(bn_layer, alpha)
            elif isinstance(bn_layer, nn.BatchNorm2d):
                bn_layer = RobustBN2d(bn_layer, alpha)
            else:
                raise RuntimeError(f"Unsupported BN layer: {name}")
            set_named_submodule(model, name, bn_layer)
        elif not isinstance(bn_layer, BN_TYPES):
            raise RuntimeError(f"Unsupported BN layer: {name}")

        bn_layer.requires_grad_(False)
        if bn_layer.weight is not None:
            bn_layer.weight.requires_grad = adapt_params == "affine"
        if bn_layer.bias is not None:
            bn_layer.bias.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("RoTTA selected no trainable parameters.")
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
    raise ValueError(f"Unknown RoTTA optimizer: {optimizer_name}")


def softmax_entropy(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return -(teacher_logits.softmax(1) * student_logits.log_softmax(1)).sum(1)


def timeliness_reweighting(ages: List[float], device: torch.device) -> torch.Tensor:
    ages_tensor = torch.tensor(ages, dtype=torch.float32, device=device)
    return torch.exp(-ages_tensor) / (1.0 + torch.exp(-ages_tensor))


def blend_timeliness_weights(ages: List[float], device: torch.device, age_loss_weight: float) -> torch.Tensor:
    strength = min(max(float(age_loss_weight), 0.0), 1.0)
    if strength <= 0:
        return torch.ones(len(ages), dtype=torch.float32, device=device)
    base = timeliness_reweighting(ages, device=device)
    return (1.0 - strength) + strength * base


def _relative_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return torch.zeros_like(x)
    scale = x.std(dim=-1, keepdim=True).clamp_min(1e-4)
    return torch.randn_like(x) * scale * noise_std


def _time_shift(x: torch.Tensor, shift_ratio: float) -> torch.Tensor:
    if shift_ratio <= 0:
        return x
    max_shift = max(int(x.shape[-1] * shift_ratio), 1)
    shifts = torch.randint(-max_shift, max_shift + 1, (x.size(0),), device=x.device)
    out = x.clone()
    for row, shift in enumerate(shifts.tolist()):
        out[row] = torch.roll(out[row], shifts=shift, dims=-1)
    return out


def _time_reverse(x: torch.Tensor, probability: float) -> torch.Tensor:
    if probability <= 0:
        return x
    mask = torch.rand(x.size(0), device=x.device) < probability
    out = x.clone()
    out[mask] = torch.flip(out[mask], dims=(-1,))
    return out


def _gaussian_blur_1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    kernel_size = 5
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size)
    channels = x.size(1)
    weight = kernel.repeat(channels, 1, 1)
    return F.conv1d(x, weight, padding=kernel_size // 2, groups=channels)


def make_rotta_strong_view(
    x: torch.Tensor,
    noise_std: float = 0.005,
    shift_ratio: float = 1.0 / 16.0,
    reverse_prob: float = 0.5,
    gain_range: Tuple[float, float] = (0.6, 1.4),
    contrast_range: Tuple[float, float] = (0.7, 1.3),
    blur_sigma_range: Tuple[float, float] = (0.1, 0.5),
) -> torch.Tensor:
    """1D analogue of RoTTA's official strong TTA transform."""
    out = x
    mean = out.mean(dim=-1, keepdim=True)
    contrast = torch.empty(out.size(0), 1, 1, device=out.device).uniform_(*contrast_range)
    gain = torch.empty(out.size(0), 1, 1, device=out.device).uniform_(*gain_range)
    out = (out - mean) * contrast + mean
    out = out * gain
    out = _time_shift(out, shift_ratio)
    out = _time_reverse(out, reverse_prob)
    sigma = float(torch.empty(1, device=out.device).uniform_(*blur_sigma_range).item())
    out = _gaussian_blur_1d(out, sigma=sigma)
    out = out + _relative_noise(out, noise_std)
    return out.clamp(-1.0, 1.0)


def make_rotta_weak_view(
    x: torch.Tensor,
    noise_std: float = DEFAULT_ROTTA_WEAK_NOISE_STD,
    shift_ratio: float = DEFAULT_ROTTA_WEAK_SHIFT_RATIO,
    gain_delta: float = DEFAULT_ROTTA_WEAK_GAIN_DELTA,
    contrast_delta: float = DEFAULT_ROTTA_WEAK_CONTRAST_DELTA,
) -> torch.Tensor:
    """Light 1D weak view for the teacher branch."""
    out = x
    mean = out.mean(dim=-1, keepdim=True)
    gain_delta = max(float(gain_delta), 0.0)
    contrast_delta = max(float(contrast_delta), 0.0)
    if contrast_delta > 0:
        contrast = torch.empty(out.size(0), 1, 1, device=out.device).uniform_(
            1.0 - contrast_delta,
            1.0 + contrast_delta,
        )
        out = (out - mean) * contrast + mean
    if gain_delta > 0:
        gain = torch.empty(out.size(0), 1, 1, device=out.device).uniform_(1.0 - gain_delta, 1.0 + gain_delta)
        out = out * gain
    out = _time_shift(out, shift_ratio)
    out = out + _relative_noise(out, noise_std)
    return out


class RoTTA(TTABase):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        lr: float = DEFAULT_ROTTA_LR,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        steps: int = DEFAULT_ROTTA_STEPS,
        protocol: Optional[str] = None,
        memory_size: int = DEFAULT_ROTTA_MEMORY_SIZE,
        update_frequency: int = DEFAULT_ROTTA_UPDATE_FREQUENCY,
        lambda_t: float = DEFAULT_ROTTA_LAMBDA_T,
        lambda_u: float = DEFAULT_ROTTA_LAMBDA_U,
        alpha: float = DEFAULT_ROTTA_ALPHA,
        nu: float = DEFAULT_ROTTA_NU,
        adapt_params: str = "affine",
        reset_each_sample: bool = False,
        adapt_mode: str = "sample",
        online_batch_size: int = DEFAULT_ROTTA_ONLINE_BATCH_SIZE,
        noise_std: float = DEFAULT_ROTTA_NOISE_STD,
        use_class_balanced_memory: bool = True,
        use_robust_bn: bool = True,
        use_timeliness: bool = True,
        age_loss_weight: float = DEFAULT_ROTTA_AGE_LOSS_WEIGHT,
        teacher_view: str = DEFAULT_ROTTA_TEACHER_VIEW,
        weak_noise_std: float = DEFAULT_ROTTA_WEAK_NOISE_STD,
        weak_shift_ratio: float = DEFAULT_ROTTA_WEAK_SHIFT_RATIO,
        weak_gain_delta: float = DEFAULT_ROTTA_WEAK_GAIN_DELTA,
        weak_contrast_delta: float = DEFAULT_ROTTA_WEAK_CONTRAST_DELTA,
    ):
        if protocol:
            protocol = protocol.lower()
            if protocol not in ROTTA_RUNTIME_PROTOCOLS:
                raise ValueError(f"protocol must be one of: {sorted(ROTTA_RUNTIME_PROTOCOLS)}")
            adapt_mode = "batch" if protocol in {"online-batch", "online-batch-bias"} else "sample"
            reset_each_sample = protocol == "standalone"
            if protocol == "online-batch-bias":
                adapt_params = "bias"
        if adapt_mode not in {"sample", "batch"}:
            raise ValueError("adapt_mode must be 'sample' or 'batch'.")
        if teacher_view not in {"identity", "weak"}:
            raise ValueError("teacher_view must be 'identity' or 'weak'.")

        super().__init__(model, device=device)
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self.optimizer_name = optimizer_name
        self.weight_decay = float(weight_decay)
        self.steps = int(steps)
        self.protocol = protocol
        self.memory_size = int(memory_size)
        self.update_frequency = int(update_frequency)
        self.lambda_t = float(lambda_t)
        self.lambda_u = float(lambda_u)
        self.alpha = float(alpha)
        self.nu = float(nu)
        self.adapt_params = adapt_params
        self.reset_each_sample = bool(reset_each_sample)
        self.adapt_mode = adapt_mode
        self.online_batch_size = max(int(online_batch_size), 1)
        self.noise_std = float(noise_std)
        self.use_class_balanced_memory = bool(use_class_balanced_memory)
        self.use_robust_bn = bool(use_robust_bn)
        self.use_timeliness = bool(use_timeliness)
        self.age_loss_weight = float(age_loss_weight) if self.use_timeliness else 0.0
        self.teacher_view = teacher_view
        self.weak_noise_std = float(weak_noise_std)
        self.weak_shift_ratio = float(weak_shift_ratio)
        self.weak_gain_delta = float(weak_gain_delta)
        self.weak_contrast_delta = float(weak_contrast_delta)
        self.current_instance = 0
        self._batch_buffer: List[torch.Tensor] = []

        self.params = configure_rotta_model(
            self.model,
            alpha=self.alpha,
            adapt_params=self.adapt_params,
            use_robust_bn=self.use_robust_bn,
        )
        self.optimizer = make_optimizer(
            self.params,
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model_ema = copy.deepcopy(self.model).to(self.device)
        for param in self.model_ema.parameters():
            param.detach_()
        memory_cls = CSTU if self.use_class_balanced_memory else GlobalCSTU
        self.mem = memory_cls(
            capacity=self.memory_size,
            num_class=self.num_classes,
            lambda_t=self.lambda_t if self.use_timeliness else 0.0,
            lambda_u=self.lambda_u,
        )

        self.initial_model_state = copy.deepcopy(self.model.state_dict())
        self.initial_ema_state = copy.deepcopy(self.model_ema.state_dict())
        self.initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.initial_memory = self.mem.clone()

    def reset(self) -> None:
        self.model.load_state_dict(self.initial_model_state, strict=True)
        self.model_ema.load_state_dict(self.initial_ema_state, strict=True)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        self.mem = self.initial_memory.clone()
        self.current_instance = 0
        self._batch_buffer = []

    def reset_for_new_sample(self) -> None:
        self.reset()

    @staticmethod
    def update_ema_variables(ema_model: nn.Module, model: nn.Module, nu: float) -> nn.Module:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1.0 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    @torch.enable_grad()
    def update_model(self) -> None:
        self.model.train()
        self.model_ema.train()
        sup_data, ages = self.mem.get_memory()
        if not sup_data:
            return
        sup_data_tensor = torch.stack(sup_data).to(self.device)
        if self.teacher_view == "weak":
            fork_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                teacher_sup_data = make_rotta_weak_view(
                    sup_data_tensor,
                    noise_std=self.weak_noise_std,
                    shift_ratio=self.weak_shift_ratio,
                    gain_delta=self.weak_gain_delta,
                    contrast_delta=self.weak_contrast_delta,
                )
        else:
            teacher_sup_data = sup_data_tensor
        strong_sup_aug = make_rotta_strong_view(sup_data_tensor, noise_std=self.noise_std)
        ema_sup_out = self.model_ema(teacher_sup_data)
        stu_sup_out = self.model(strong_sup_aug)
        instance_weight = blend_timeliness_weights(
            ages,
            device=self.device,
            age_loss_weight=self.age_loss_weight,
        )
        loss = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @torch.no_grad()
    def _teacher_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.model_ema.eval()
        return self.model_ema(inputs)

    def forward_and_adapt(self, batch_data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            ema_out = self._teacher_predict(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)

            for idx, data in enumerate(batch_data):
                current_instance = (data.detach(), int(pseudo_label[idx].item()), float(entropy[idx].item()))
                self.mem.add_instance(current_instance)
                self.current_instance += 1
                if self.current_instance % self.update_frequency == 0:
                    break_update = True
                else:
                    break_update = False
                if break_update:
                    # Leave no_grad before the student update.
                    pass

        if self.current_instance % self.update_frequency == 0:
            self.update_model()
        return ema_out.detach()

    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.reset_each_sample:
            self.reset()

        if self.adapt_mode == "batch":
            logits_parts = []
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


def run_rotta_evaluation(
    loaders,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
    num_classes: int,
    protocol: str = "online-batch",
    lr: float = DEFAULT_ROTTA_LR,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    steps: int = DEFAULT_ROTTA_STEPS,
    memory_size: int = DEFAULT_ROTTA_MEMORY_SIZE,
    update_frequency: int = DEFAULT_ROTTA_UPDATE_FREQUENCY,
    lambda_t: float = DEFAULT_ROTTA_LAMBDA_T,
    lambda_u: float = DEFAULT_ROTTA_LAMBDA_U,
    alpha: float = DEFAULT_ROTTA_ALPHA,
    nu: float = DEFAULT_ROTTA_NU,
    online_batch_size: int = DEFAULT_ROTTA_ONLINE_BATCH_SIZE,
    noise_std: float = DEFAULT_ROTTA_NOISE_STD,
    use_class_balanced_memory: bool = True,
    use_robust_bn: bool = True,
    use_timeliness: bool = True,
    age_loss_weight: float = DEFAULT_ROTTA_AGE_LOSS_WEIGHT,
    teacher_view: str = DEFAULT_ROTTA_TEACHER_VIEW,
    weak_noise_std: float = DEFAULT_ROTTA_WEAK_NOISE_STD,
    weak_shift_ratio: float = DEFAULT_ROTTA_WEAK_SHIFT_RATIO,
    weak_gain_delta: float = DEFAULT_ROTTA_WEAK_GAIN_DELTA,
    weak_contrast_delta: float = DEFAULT_ROTTA_WEAK_CONTRAST_DELTA,
) -> float:
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    rotta = RoTTA(
        model=model,
        device=device,
        num_classes=num_classes,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        steps=steps,
        protocol=protocol,
        memory_size=memory_size,
        update_frequency=update_frequency,
        lambda_t=lambda_t,
        lambda_u=lambda_u,
        alpha=alpha,
        nu=nu,
        online_batch_size=online_batch_size,
        noise_std=noise_std,
        use_class_balanced_memory=use_class_balanced_memory,
        use_robust_bn=use_robust_bn,
        use_timeliness=use_timeliness,
        age_loss_weight=age_loss_weight,
        teacher_view=teacher_view,
        weak_noise_std=weak_noise_std,
        weak_shift_ratio=weak_shift_ratio,
        weak_gain_delta=weak_gain_delta,
        weak_contrast_delta=weak_contrast_delta,
    )
    return float(rotta.predict_loader(loaders["test"]).get("acc", 0.0))


def run_rotta_store_evaluation(
    loaders,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
    num_classes: int,
    lr: float = DEFAULT_ROTTA_LR,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    steps: int = DEFAULT_ROTTA_STEPS,
    memory_size: int = DEFAULT_ROTTA_MEMORY_SIZE,
    update_frequency: int = DEFAULT_ROTTA_UPDATE_FREQUENCY,
    lambda_t: float = DEFAULT_ROTTA_LAMBDA_T,
    lambda_u: float = DEFAULT_ROTTA_LAMBDA_U,
    alpha: float = DEFAULT_ROTTA_ALPHA,
    nu: float = DEFAULT_ROTTA_NU,
    online_batch_size: int = DEFAULT_ROTTA_ONLINE_BATCH_SIZE,
    noise_std: float = DEFAULT_ROTTA_NOISE_STD,
    use_class_balanced_memory: bool = True,
    use_robust_bn: bool = True,
    use_timeliness: bool = True,
    age_loss_weight: float = DEFAULT_ROTTA_AGE_LOSS_WEIGHT,
    teacher_view: str = DEFAULT_ROTTA_TEACHER_VIEW,
    weak_noise_std: float = DEFAULT_ROTTA_WEAK_NOISE_STD,
    weak_shift_ratio: float = DEFAULT_ROTTA_WEAK_SHIFT_RATIO,
    weak_gain_delta: float = DEFAULT_ROTTA_WEAK_GAIN_DELTA,
    weak_contrast_delta: float = DEFAULT_ROTTA_WEAK_CONTRAST_DELTA,
) -> float:
    """Evaluate RoTTA with replayed seen-store phases.

    At phase k, reset RoTTA to the source state, replay all chunks seen so far
    through the normal RoTTA adaptation pipeline, then evaluate only the current
    chunk with the adapted EMA teacher. This mirrors the earlier BNAdapt
    protocol1 semantics.
    """
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    rotta = RoTTA(
        model=model,
        device=device,
        num_classes=num_classes,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        steps=steps,
        protocol="online-batch",
        memory_size=memory_size,
        update_frequency=update_frequency,
        lambda_t=lambda_t,
        lambda_u=lambda_u,
        alpha=alpha,
        nu=nu,
        online_batch_size=online_batch_size,
        noise_std=noise_std,
        use_class_balanced_memory=use_class_balanced_memory,
        use_robust_bn=use_robust_bn,
        use_timeliness=use_timeliness,
        age_loss_weight=age_loss_weight,
        teacher_view=teacher_view,
        weak_noise_std=weak_noise_std,
        weak_shift_ratio=weak_shift_ratio,
        weak_gain_delta=weak_gain_delta,
        weak_contrast_delta=weak_contrast_delta,
    )

    chunks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    total_correct = 0
    total_samples = 0

    for inputs, labels in loaders["test"]:
        inputs = torch.from_numpy(inputs) if isinstance(inputs, np.ndarray) else inputs
        labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
        inputs = inputs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        chunks.append((inputs, labels))

        seen_inputs = torch.cat([chunk_inputs for chunk_inputs, _ in chunks], dim=0)
        rotta.reset()
        rotta.adapt_one_batch(seen_inputs)

        rotta.model_ema.eval()
        with torch.no_grad():
            logits = rotta.model_ema(inputs)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_correct / max(total_samples, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQ RoTTA evaluation")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="checkpoints/resnet18_sq_clean_noaug.pth")
    parser.add_argument("--test_speeds", type=str, default="2,3")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--protocol", type=str, default="online-batch", choices=sorted(ROTTA_PROTOCOLS))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_speeds", type=str, default="0,1")
    parser.add_argument("--corruption_type", type=str, default=None, choices=["noise", "missing"])
    parser.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--dataset_transform", action="store_true")
    parser.add_argument("--no_model_transform", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--lr", type=float, default=DEFAULT_ROTTA_LR)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=DEFAULT_ROTTA_STEPS)
    parser.add_argument("--memory_size", type=int, default=DEFAULT_ROTTA_MEMORY_SIZE)
    parser.add_argument("--update_frequency", type=int, default=DEFAULT_ROTTA_UPDATE_FREQUENCY)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ROTTA_ALPHA)
    parser.add_argument("--nu", type=float, default=DEFAULT_ROTTA_NU)
    parser.add_argument("--lambda_t", type=float, default=DEFAULT_ROTTA_LAMBDA_T)
    parser.add_argument("--lambda_u", type=float, default=DEFAULT_ROTTA_LAMBDA_U)
    parser.add_argument("--noise_std", type=float, default=DEFAULT_ROTTA_NOISE_STD)
    parser.add_argument("--age_loss_weight", type=float, default=DEFAULT_ROTTA_AGE_LOSS_WEIGHT)
    parser.add_argument("--teacher_view", type=str, default=DEFAULT_ROTTA_TEACHER_VIEW, choices=["identity", "weak"])
    parser.add_argument("--weak_noise_std", type=float, default=DEFAULT_ROTTA_WEAK_NOISE_STD)
    parser.add_argument("--weak_shift_ratio", type=float, default=DEFAULT_ROTTA_WEAK_SHIFT_RATIO)
    parser.add_argument("--weak_gain_delta", type=float, default=DEFAULT_ROTTA_WEAK_GAIN_DELTA)
    parser.add_argument("--weak_contrast_delta", type=float, default=DEFAULT_ROTTA_WEAK_CONTRAST_DELTA)
    parser.add_argument("--no_class_balanced_memory", action="store_true")
    parser.add_argument("--no_robust_bn", action="store_true")
    parser.add_argument("--no_timeliness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    loader_cfg = LoaderConfig(batch_size=args.batch_size, shuffle_test=False)
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
        "memory_size": args.memory_size,
        "update_frequency": args.update_frequency,
        "lambda_t": args.lambda_t,
        "lambda_u": args.lambda_u,
        "alpha": args.alpha,
        "nu": args.nu,
        "online_batch_size": args.batch_size,
        "noise_std": args.noise_std,
        "use_class_balanced_memory": not args.no_class_balanced_memory,
        "use_robust_bn": not args.no_robust_bn,
        "use_timeliness": not args.no_timeliness,
        "age_loss_weight": args.age_loss_weight,
        "teacher_view": args.teacher_view,
        "weak_noise_std": args.weak_noise_std,
        "weak_shift_ratio": args.weak_shift_ratio,
        "weak_gain_delta": args.weak_gain_delta,
        "weak_contrast_delta": args.weak_contrast_delta,
    }
    if args.protocol == "online-batch-store":
        acc = run_rotta_store_evaluation(**common_kwargs)
    else:
        acc = run_rotta_evaluation(protocol=args.protocol, **common_kwargs)
    print(f"Baseline Acc: {float(baseline.get('acc', 0.0)):.4%}")
    print(f"RoTTA Acc: {acc:.4%}")


if __name__ == "__main__":
    main()
