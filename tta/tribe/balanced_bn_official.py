"""Official TRIBE balanced BN update (Gorilla-Lab-SCUT/TRIBE balanced_bn_pyv)."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn


def update_statistics_1d_v5(
    local_mean: torch.Tensor,
    local_var: torch.Tensor,
    global_mean: torch.Tensor,
    global_var: torch.Tensor,
    momentum: float,
    data: torch.Tensor,
    label: torch.Tensor,
    gamma: float,
    training: bool = False,
) -> None:
    if local_mean.size(0) != local_var.size(0):
        raise RuntimeError("the sizes of local_mean and local_var are inequal.")

    if data.size(0) != label.size(0):
        raise RuntimeError("the values of the first dimension of data and label are inequal.")

    if not training:
        return

    unique = label.unique()
    label_mapping = torch.arange(unique.size(0), device=label.device)
    reverse = label.new_full((local_mean.size(0),), -1)
    reverse[unique] = label_mapping
    label_local = reverse[label]
    lm = local_mean[label]

    label_num = torch.zeros_like(label_mapping)
    label_num.scatter_add_(0, label_local, torch.ones_like(label_local))
    mask = label_num > (1.0 / momentum)
    m = torch.where(mask, 1.0 / label_num.float(), momentum)

    delta_pre = data - lm[..., None]
    delta_k = delta_pre.new_zeros((unique.size(0), delta_pre.size(1)))
    delta_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.mean(2))
    delta_k *= m.view(-1, 1)
    local_mean[unique] = (1.0 - gamma) * delta_k + local_mean[unique]
    local_mean.add_(gamma * delta_k.mean(0, keepdim=True))

    delta_square_k = delta_pre.new_zeros((unique.size(0), delta_pre.size(1)))
    delta_square_k.scatter_add_(
        0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.pow(2).mean(2)
    )
    local_var[unique] = local_var[unique] + (1.0 - gamma) * (
        m.view(-1, 1) * (delta_square_k - label_num.view(-1, 1) * local_var[unique]) - delta_k.pow(2)
    )

    var_gap = (m.view(-1, 1) * delta_square_k - delta_k.pow(2)).mean(0, keepdim=True)
    local_var.add_(gamma * (var_gap - (m * label_num).mean() * local_var))

    global_mean.copy_(local_mean.mean(0))
    global_var.copy_(local_var.mean(0) + local_mean.var(0, unbiased=False))


class BalancedRobustBN1dOfficial(nn.Module):
    """Conv1d [B, C, L] balanced BN matching official BalancedRobustBN1dV5."""

    def __init__(
        self,
        bn_layer: nn.BatchNorm1d,
        num_classes: int,
        momentum: float,
        gamma: float,
    ) -> None:
        super().__init__()
        if bn_layer.running_mean is None or bn_layer.running_var is None:
            raise ValueError("TRIBE BalancedBN requires source BatchNorm running statistics.")

        self.num_features = int(bn_layer.num_features)
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.gamma = float(gamma)
        self.eps = float(bn_layer.eps)

        self.register_buffer("global_mean", deepcopy(bn_layer.running_mean))
        self.register_buffer("global_var", deepcopy(bn_layer.running_var))
        local_mean = bn_layer.running_mean.detach().clone()[None, ...].expand(num_classes, -1).clone()
        local_var = bn_layer.running_var.detach().clone()[None, ...].expand(num_classes, -1).clone()
        self.register_buffer("local_mean", local_mean)
        self.register_buffer("local_var", local_var)

        if bn_layer.weight is not None:
            self.weight = nn.Parameter(bn_layer.weight.detach().clone())
        else:
            self.register_parameter("weight", None)
        if bn_layer.bias is not None:
            self.bias = nn.Parameter(bn_layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.label: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.global_mean.detach_()
        self.global_var.detach_()
        self.local_mean.detach_()
        self.local_var.detach_()

        label = self.label
        if label is not None:
            update_statistics_1d_v5(
                self.local_mean,
                self.local_var,
                self.global_mean,
                self.global_var,
                self.momentum,
                x,
                label,
                self.gamma,
                self.training,
            )
            self.label = None
        elif self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2], unbiased=False, keepdim=False)
            self.global_mean.mul_(1.0 - self.momentum).add_(b_mean, alpha=self.momentum)
            self.global_var.mul_(1.0 - self.momentum).add_(b_var, alpha=self.momentum)

        out = (x - self.global_mean[None, :, None]) / torch.sqrt(self.global_var[None, :, None] + self.eps)
        if self.weight is not None:
            out = self.weight[None, :, None] * out
        if self.bias is not None:
            out = out + self.bias[None, :, None]
        return out


class BalancedRobustBN2dOfficial(nn.Module):
    """BatchNorm2d balanced BN matching official BalancedRobustBN2dV5 update path."""

    def __init__(
        self,
        bn_layer: nn.BatchNorm2d,
        num_classes: int,
        momentum: float,
        gamma: float,
    ) -> None:
        super().__init__()
        if bn_layer.running_mean is None or bn_layer.running_var is None:
            raise ValueError("TRIBE BalancedBN requires source BatchNorm running statistics.")

        self.num_features = int(bn_layer.num_features)
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.gamma = float(gamma)
        self.eps = float(bn_layer.eps)

        self.register_buffer("global_mean", deepcopy(bn_layer.running_mean))
        self.register_buffer("global_var", deepcopy(bn_layer.running_var))
        local_mean = bn_layer.running_mean.detach().clone()[None, ...].expand(num_classes, -1).clone()
        local_var = bn_layer.running_var.detach().clone()[None, ...].expand(num_classes, -1).clone()
        self.register_buffer("local_mean", local_mean)
        self.register_buffer("local_var", local_var)

        if bn_layer.weight is not None:
            self.weight = nn.Parameter(bn_layer.weight.detach().clone())
        else:
            self.register_parameter("weight", None)
        if bn_layer.bias is not None:
            self.bias = nn.Parameter(bn_layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        self.label: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.global_mean.detach_()
        self.global_var.detach_()
        self.local_mean.detach_()
        self.local_var.detach_()

        label = self.label
        if label is not None:
            self._update_with_labels(x, label)
            self.label = None
        elif self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)
            self.global_mean.mul_(1.0 - self.momentum).add_(b_mean, alpha=self.momentum)
            self.global_var.mul_(1.0 - self.momentum).add_(b_var, alpha=self.momentum)

        out = (x - self.global_mean[None, :, None, None]) / torch.sqrt(
            self.global_var[None, :, None, None] + self.eps
        )
        if self.weight is not None:
            out = self.weight[None, :, None, None] * out
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out

    def _update_with_labels(self, data: torch.Tensor, label: torch.Tensor) -> None:
        unique = label.unique()
        label_mapping = torch.arange(unique.size(0), device=label.device)
        reverse = label.new_full((self.local_mean.size(0),), -1)
        reverse[unique] = label_mapping
        label_local = reverse[label]
        lm = self.local_mean[label]

        label_num = torch.zeros_like(label_mapping)
        label_num.scatter_add_(0, label_local, torch.ones_like(label_local))
        mask = label_num > (1.0 / self.momentum)
        m = torch.where(mask, 1.0 / label_num.float(), self.momentum)

        delta_pre = data - lm[..., None, None]
        delta_k = delta_pre.new_zeros((unique.size(0), delta_pre.size(1)))
        delta_k.scatter_add_(
            0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.mean((2, 3))
        )
        delta_k *= m.view(-1, 1)
        self.local_mean[unique] = (1.0 - self.gamma) * delta_k + self.local_mean[unique]
        self.local_mean.add_(self.gamma * delta_k.mean(0, keepdim=True))

        delta_square_k = delta_pre.new_zeros((unique.size(0), delta_pre.size(1)))
        delta_square_k.scatter_add_(
            0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.pow(2).mean((2, 3))
        )
        self.local_var[unique] = self.local_var[unique] + (1.0 - self.gamma) * (
            m.view(-1, 1) * (delta_square_k - label_num.view(-1, 1) * self.local_var[unique])
            - delta_k.pow(2)
        )

        var_gap = (m.view(-1, 1) * delta_square_k - delta_k.pow(2)).mean(0, keepdim=True)
        self.local_var.add_(self.gamma * (var_gap - (m * label_num).mean() * self.local_var))

        self.global_mean.copy_(self.local_mean.mean(0))
        self.global_var.copy_(self.local_var.mean(0) + self.local_mean.var(0, unbiased=False))
