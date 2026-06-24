"""Sequence-spectrogram fusion network for 1D machinery signals.

This follows the SSFN paper structure: a sequence branch, a spectrogram
branch, one fusion connection with HLFA, and an HLFA-equipped second sequence
block.  The public forward accepts raw signals shaped ``(N, C, L)`` so it can
be used by the existing 1D TTA/TTT code.
"""

import math
import sys
from typing import Iterable, Tuple

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ts_transform import transform_value


def _conv_out_length(length: int, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> int:
    return (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class PreActResidual1D(nn.Module):
    def __init__(self, channels: int, gamma: int = 4, kernel_size: int = 7, padding: int = 3):
        super().__init__()
        hidden = int(channels) * int(gamma)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channels, hidden, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + x


class PreActResidual2D(nn.Module):
    def __init__(self, channels: int, gamma: int = 4, kernel_size: int = 7, padding: int = 3):
        super().__init__()
        hidden = int(channels) * int(gamma)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + x


class HybridLevelFeatureAggregator2D(nn.Module):
    """HLFA with local convolution and multi-head global self-attention."""

    def __init__(
        self,
        channels: int,
        dk: int,
        dv: int,
        num_heads: int,
        local_kernel_size: Tuple[int, int] = (3, 3),
    ):
        super().__init__()
        if dk % num_heads != 0 or dv % num_heads != 0:
            raise ValueError("dk and dv must be divisible by num_heads")
        if dk >= channels:
            raise ValueError("HLFA requires dk < channels so the local branch has channels")
        self.channels = int(channels)
        self.dk = int(dk)
        self.dv = int(dv)
        self.num_heads = int(num_heads)
        local_channels = self.channels - self.dk
        padding = (local_kernel_size[0] // 2, local_kernel_size[1] // 2)
        self.local = nn.Sequential(
            nn.Conv2d(self.channels, local_channels, kernel_size=local_kernel_size, padding=padding, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.q_proj = nn.Conv2d(self.channels, self.dk, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(self.channels, self.dk, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(self.channels, self.dv, kernel_size=1, bias=False)
        self.attn_out = nn.Conv2d(self.dv, self.dk, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        tokens = height * width
        local = self.local(x)

        q = self.q_proj(x).reshape(batch_size, self.num_heads, self.dk // self.num_heads, tokens)
        k = self.k_proj(x).reshape(batch_size, self.num_heads, self.dk // self.num_heads, tokens)
        v = self.v_proj(x).reshape(batch_size, self.num_heads, self.dv // self.num_heads, tokens)

        q = q.transpose(2, 3)
        scale = math.sqrt(max(self.dk // self.num_heads, 1))
        attn = torch.matmul(q, k) / scale
        attn = torch.softmax(attn, dim=-1)
        value = torch.matmul(attn, v.transpose(2, 3))
        value = value.transpose(2, 3).reshape(batch_size, self.dv, height, width)
        global_feat = self.attn_out(value)
        return torch.cat([global_feat, local], dim=1)


class HybridLevelFeatureAggregator1D(nn.Module):
    """1D realization of the sequence-block HLFA after the paper's reshape."""

    def __init__(self, channels: int, dk: int, dv: int, num_heads: int, local_kernel_size: int = 3):
        super().__init__()
        if dk % num_heads != 0 or dv % num_heads != 0:
            raise ValueError("dk and dv must be divisible by num_heads")
        if dk >= channels:
            raise ValueError("HLFA requires dk < channels so the local branch has channels")
        self.channels = int(channels)
        self.dk = int(dk)
        self.dv = int(dv)
        self.num_heads = int(num_heads)
        local_channels = self.channels - self.dk
        self.local = nn.Sequential(
            nn.Conv1d(self.channels, local_channels, kernel_size=local_kernel_size, padding=local_kernel_size // 2, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.q_proj = nn.Conv1d(self.channels, self.dk, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(self.channels, self.dk, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(self.channels, self.dv, kernel_size=1, bias=False)
        self.attn_out = nn.Conv1d(self.dv, self.dk, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = x.shape
        local = self.local(x)
        q = self.q_proj(x).reshape(batch_size, self.num_heads, self.dk // self.num_heads, length)
        k = self.k_proj(x).reshape(batch_size, self.num_heads, self.dk // self.num_heads, length)
        v = self.v_proj(x).reshape(batch_size, self.num_heads, self.dv // self.num_heads, length)

        q = q.transpose(2, 3)
        scale = math.sqrt(max(self.dk // self.num_heads, 1))
        attn = torch.matmul(q, k) / scale
        attn = torch.softmax(attn, dim=-1)
        value = torch.matmul(attn, v.transpose(2, 3))
        value = value.transpose(2, 3).reshape(batch_size, self.dv, length)
        global_feat = self.attn_out(value)
        return torch.cat([global_feat, local], dim=1)


class SpectrogramToSequenceMapper(nn.Module):
    """Map HLFA spectrogram features to the sequence branch length.

    The paper describes flatten + FC + BN + ReLU.  For the longer SQ windows,
    this resolution-adaptive form keeps the same mapping role without making
    the FC layer dominate the compact SSFN parameter budget.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        x = x.flatten(2)
        x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return self.relu(self.bn(self.proj(x)))


class SSFN(nn.Module):
    def __init__(
        self,
        input_length: int = 2048,
        num_classes: int = 7,
        input_channels: int = 1,
        name: str = "ssfn",
        transform_in_model: bool = True,
        zero_mean: bool = True,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        base_channels: int = 20,
        kernel_size: int = 7,
        conv_stride: int = 2,
        padding: int = 3,
        gamma: int = 4,
        stft_n_fft: int = 128,
        stft_win_length: int = 60,
        stft_hop_length: int = 54,
        spectrogram_kind: str = "stft",
        cwt_kernel_size: int = 129,
        normalize_spectrogram: bool = True,
        spec_erasing: bool = True,
        erasing_ratios: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    ):
        super().__init__()
        if spectrogram_kind not in {"stft", "cwt"}:
            raise ValueError("spectrogram_kind must be 'stft' or 'cwt'")
        self.name = name
        self.input_length = int(input_length)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.transform_in_model = bool(transform_in_model)
        self.zero_mean = bool(zero_mean)
        self.clamp_range = clamp_range
        self.base_channels = int(base_channels)
        self.spectrogram_kind = spectrogram_kind
        self.n_fft = int(stft_n_fft)
        self.win_length = int(stft_win_length)
        self.hop_length = int(stft_hop_length)
        self.cwt_kernel_size = int(cwt_kernel_size)
        self.normalize_spectrogram = bool(normalize_spectrogram)
        self.spec_erasing = bool(spec_erasing)
        self.erasing_ratios = tuple(float(x) for x in erasing_ratios)
        self.feature_dim = self.base_channels * 2
        self.register_buffer("stft_window", torch.hamming_window(self.win_length), persistent=False)

        self.seq_conv0 = nn.Conv1d(
            self.input_channels,
            self.base_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=padding,
            bias=False,
        )
        self.seq_bn0 = nn.BatchNorm1d(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.seq_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.seq_block1 = PreActResidual1D(self.base_channels, gamma=gamma, kernel_size=kernel_size, padding=padding)

        self.spec_conv0 = nn.Conv2d(
            self.input_channels,
            self.base_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=padding,
            bias=False,
        )
        self.spec_bn0 = nn.BatchNorm2d(self.base_channels)
        self.spec_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.spec_block1 = PreActResidual2D(self.base_channels, gamma=gamma, kernel_size=kernel_size, padding=padding)
        self.spec_hlfa1 = HybridLevelFeatureAggregator2D(self.base_channels, dk=18, dv=18, num_heads=2)
        self.spec_mapper1 = SpectrogramToSequenceMapper(self.base_channels)

        fused_channels = self.base_channels * 2
        self.seq_hlfa2 = HybridLevelFeatureAggregator1D(fused_channels, dk=2, dv=2, num_heads=1)
        self.seq_block2 = PreActResidual1D(fused_channels, gamma=gamma, kernel_size=kernel_size, padding=padding)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(fused_channels, self.num_classes)

        conv_len = _conv_out_length(self.input_length, kernel_size, conv_stride, padding)
        self.sequence_feature_length = _conv_out_length(conv_len, 3, 2, 1)
        self.spectrogram_size = (self.n_fft // 2 + 1, 1 + self.input_length // self.hop_length)
        self.weights_init()

    def weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def transform_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize_signal(x)

    def normalize_signal(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.transform_in_model:
            x = transform_value(x, zero_mean=self.zero_mean)
        else:
            x = x - x.mean(dim=-1, keepdim=True)
        return torch.clamp(x, *self.clamp_range)

    def _stft_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        signal = x.reshape(batch_size * channels, length)
        window = self.stft_window.to(device=signal.device, dtype=signal.dtype)
        spec = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = torch.log10(spec.abs().clamp_min(1e-6))
        return mag.reshape(batch_size, channels, mag.size(-2), mag.size(-1))

    def _cwt_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        signal = x.reshape(batch_size * channels, 1, length)
        freq_bins, time_bins = self.spectrogram_size
        kernel_size = min(self.cwt_kernel_size, length if length % 2 == 1 else length - 1)
        half_width = max(kernel_size // 2, 1)
        t = torch.linspace(-3.0, 3.0, kernel_size, device=x.device, dtype=x.dtype)
        scales = torch.linspace(1.0, 32.0, freq_bins, device=x.device, dtype=x.dtype).view(-1, 1)
        wave = torch.exp(-0.5 * (t.view(1, -1) / scales).pow(2))
        wave = wave * torch.cos(5.0 * t.view(1, -1) / scales)
        wave = wave - wave.mean(dim=1, keepdim=True)
        wave = wave / wave.norm(dim=1, keepdim=True).clamp_min(1e-6)
        coeff = F.conv1d(signal, wave.unsqueeze(1), padding=half_width).abs()
        coeff = coeff[:, :, :length]
        coeff = F.interpolate(coeff.unsqueeze(1), size=(freq_bins, time_bins), mode="bilinear", align_corners=False)
        coeff = coeff.squeeze(1)
        mag = torch.log10(coeff.clamp_min(1e-6))
        return mag.reshape(batch_size, channels, freq_bins, time_bins)

    def signal_to_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize_signal(x)
        if self.spectrogram_kind == "cwt":
            spec = self._cwt_spectrogram(x)
        else:
            spec = self._stft_spectrogram(x)
        if self.normalize_spectrogram:
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            spec = (spec - mean) / std
        spec = torch.clamp(spec, -5.0, 5.0)
        if self.training and self.spec_erasing:
            spec = self.random_erase_spectrogram(spec)
        return spec

    def random_erase_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.erasing_ratios:
            return spec
        out = spec.clone()
        batch_size, _, height, width = out.shape
        if height < 2 or width < 2:
            return out
        for row in range(batch_size):
            ratio_idx = int(torch.randint(0, len(self.erasing_ratios), (1,), device=out.device).item())
            ratio = self.erasing_ratios[ratio_idx]
            erase_h_min = max(int(0.1 * height), 1)
            erase_w_min = max(int(0.1 * width), 1)
            erase_h_max = max(min(int(ratio * height), height), erase_h_min)
            erase_w_max = max(min(int(ratio * width), width), erase_w_min)
            erase_h = int(torch.randint(erase_h_min, erase_h_max + 1, (1,), device=out.device).item())
            erase_w = int(torch.randint(erase_w_min, erase_w_max + 1, (1,), device=out.device).item())
            top = int(torch.randint(0, max(height - erase_h + 1, 1), (1,), device=out.device).item())
            left = int(torch.randint(0, max(width - erase_w + 1, 1), (1,), device=out.device).item())
            noise = torch.empty_like(out[row : row + 1, :, top : top + erase_h, left : left + erase_w])
            noise.uniform_(-1.0, 1.0)
            out[row : row + 1, :, top : top + erase_h, left : left + erase_w] = noise
        return out

    def features(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.normalize_signal(x)
        seq = self.seq_pool(self.relu(self.seq_bn0(self.seq_conv0(signal))))
        seq = self.seq_block1(seq)

        spec = self.signal_to_spectrogram(x)
        spec = self.spec_pool(self.relu(self.spec_bn0(self.spec_conv0(spec))))
        spec = self.spec_block1(spec)
        spec_fused = self.spec_hlfa1(spec)
        spec_to_seq = self.spec_mapper1(spec_fused, target_length=seq.size(-1))

        fused = torch.cat([seq, spec_to_seq], dim=1)
        fused = self.seq_hlfa2(fused)
        fused = self.seq_block2(fused)
        pooled = self.avgpool(fused)
        return torch.flatten(pooled, 1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.features(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)


def ssfn(num_classes=10, input_channels=1, name="ssfn", input_length=2048, **kwargs):
    return SSFN(
        input_length=input_length,
        num_classes=num_classes,
        input_channels=input_channels,
        name=name,
        **kwargs,
    )
