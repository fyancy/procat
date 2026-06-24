"""TFN: Time-Frequency Network (Chen et al., embedded STTF layer).

Ported from https://github.com/ChenQian0618/TFN (TFN_STTF variant).
Adapted for SQ TTA pipeline: 2048-point input, model-side transform, PeTTA hooks.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.ts_transform import transform_value

# Official TFN frequency bounds for STTF / Chirplet kernels.
_FMIN = 0.03
_FMAX = 0.45


class BaseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.stride = stride
        self.padding = padding
        self.phases = ("real", "imag")
        self.weight = torch.Tensor(len(self.phases), out_channels, in_channels, self.kernel_size)
        if bias:
            self.bias = torch.Tensor(len(self.phases), out_channels)
        else:
            self.bias = None

        for phase in self.phases:
            idx = self.phases.index(phase)
            self.weight[idx] = torch.Tensor(out_channels, in_channels, self.kernel_size)
            init.kaiming_uniform_(self.weight[idx], a=math.sqrt(5))
            if bias:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[idx])
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias[idx], -bound, bound)

        if self.__class__.__name__ == "BaseConv1d":
            self.weight = nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = []
        for phase in self.phases:
            idx = self.phases.index(phase)
            outputs.append(
                F.conv1d(
                    input,
                    self.weight[idx],
                    bias=None if self.bias is None else self.bias[idx],
                    stride=self.stride,
                    padding=self.padding,
                )
            )
        return torch.sqrt(outputs[0].pow(2) + outputs[1].pow(2))


class BaseFuncConv1d(BaseConv1d):
    def __init__(self, *pargs, **kwargs) -> None:
        kwargs_new = {
            k: kwargs[k]
            for k in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "bias")
            if k in kwargs
        }
        super().__init__(*pargs, **kwargs_new)
        if self.__class__.__name__ == "BaseFuncConv1d":
            self.weight = nn.Parameter(self.weight)
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias)
            self.superparams = self.weight

    def _clamp_parameters(self) -> None:
        with torch.no_grad():
            for i, (lo, hi) in enumerate(self.params_bound):
                self.superparams.data[:, :, i].clamp_(lo, hi)

    def WeightForward(self) -> None:
        if self.clamp_flag:
            self._clamp_parameters()
        stacks = []
        for phase in self.phases:
            per_out = []
            for i in range(self.superparams.shape[0]):
                per_in = []
                for j in range(self.superparams.shape[1]):
                    per_in.append(
                        self.weightforward(self.kernel_size, self.superparams[i, j], phase).unsqueeze(0)
                    )
                per_out.append(torch.vstack(per_in).unsqueeze(0))
            stacks.append(torch.vstack(per_out).unsqueeze(0))
        self.weight = torch.vstack(stacks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.__class__.__name__ != "BaseFuncConv1d":
            self.WeightForward()
        return super().forward(input)


class TFconv_STTF(BaseFuncConv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        clamp_flag: bool = True,
        params_bound: Tuple[Tuple[float, float], ...] = ((0.0, 0.5),),
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.clamp_flag = clamp_flag
        self.params_bound = params_bound
        self.superparams = nn.Parameter(torch.Tensor(out_channels, in_channels, len(params_bound)))
        self._reset_parameters()
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias)

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            temp0 = torch.linspace(_FMIN, _FMAX, shape.numel()).reshape(shape)
            self.superparams.data[:, :, 0] = temp0
            self.WeightForward()

    def weightforward(self, lens, params, phase: str) -> torch.Tensor:
        if isinstance(lens, torch.Tensor):
            lens = int(lens.item())
        t = torch.arange(-(lens // 2), lens - (lens // 2), device=params.device)
        sigma = torch.tensor(0.52, device=params.device)
        envelope = torch.exp(-(t / (lens // 2)).pow(2) / sigma.pow(2) / 2)
        if self.phases.index(phase) == 0:
            return envelope * torch.cos(2 * math.pi * params[0] * t)
        return envelope * torch.sin(2 * math.pi * params[0] * t)


class TFNBackboneCNN(nn.Module):
    """Backbone CNN from the official TFN repository."""

    def __init__(self, in_channels: int = 1, out_channels: int = 10, kernel_size: int = 15) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=kernel_size, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64, out_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = torch.squeeze(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.layer5(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encode(x))


class TFN_STTF(nn.Module):
    """TFN with learnable STTF preprocessing layer (official default variant)."""

    def __init__(
        self,
        input_length: int = 2048,
        num_classes: int = 7,
        mid_channel: int = 32,
        kernel_size: int = 11,
        clamp_flag: bool = True,
        transform_in_model: bool = False,
        zero_mean: bool = True,
        in_channels: int = 1,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        del input_length  # backbone uses adaptive pooling; length only documents protocol.
        self.transform_in_model = transform_in_model
        self.zero_mean = zero_mean
        self.clamp_range = clamp_range
        self.track_running_stats = track_running_stats
        self.name = "tfn_sttf"

        funckernel_size = mid_channel * 2 - 1
        self.backbone = TFNBackboneCNN(in_channels=mid_channel, out_channels=num_classes, kernel_size=kernel_size)
        self.funconv = TFconv_STTF(
            in_channels,
            mid_channel,
            funckernel_size,
            padding=funckernel_size // 2,
            bias=False,
            clamp_flag=clamp_flag,
        )
        self._set_bn_track_running_stats(track_running_stats)

    def _set_bn_track_running_stats(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = enabled

    def transform_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_in_model:
            x = transform_value(x, zero_mean=self.zero_mean)
        x = x - x.mean(dim=-1, keepdim=True)
        return torch.clamp(x, *self.clamp_range)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform_fn(x)
        x = self.funconv(x)
        return self.backbone.encode(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self._encode(x))


def tfn_sttf(**kwargs) -> TFN_STTF:
    return TFN_STTF(**kwargs)
