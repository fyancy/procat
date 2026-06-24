"""ViTTT-style spectrogram backbone for SQ TTT2D experiments."""

import sys
from typing import Tuple

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ts_transform import transform_value


class ViTTTMixer(nn.Module):
    """Visual TTT token mixer from ViT^3, adapted for spectrogram tokens."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        head_dim = dim // num_heads
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)
        nn.init.trunc_normal_(self.w3, std=0.02)
        self.proj = nn.Linear(dim + head_dim, dim)
        self.scale = 9**-0.5

    def inner_train_simplified_swiglu(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        lr: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z1 = k @ w1
        z2 = k @ w2
        sig = torch.sigmoid(z2)
        a = z2 * sig
        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - lr * g1, w2 - lr * g2

    def inner_train_3x3dwc(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        lr: float = 1.0,
    ) -> torch.Tensor:
        batch_size, channels, height, width = k.shape
        e = -v / float(v.shape[2] * v.shape[3]) * self.scale
        padded = F.pad(k, (1, 1, 1, 1))
        outs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys = 1 + dy
                xs = 1 + dx
                dot = (padded[:, :, ys : ys + height, xs : xs + width] * e).sum(dim=(-2, -1))
                outs.append(dot)
        grad = torch.stack(outs, dim=-1).reshape(batch_size * channels, 1, 3, 3)
        grad = grad / (grad.norm(dim=(-2, -1), keepdim=True) + 1.0)
        return w.repeat(batch_size, 1, 1, 1) - lr * grad

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        head_dim = channels // self.num_heads
        q1, k1, v1, q2, k2, v2 = torch.split(
            self.qkv(x),
            [channels, channels, channels, head_dim, head_dim, head_dim],
            dim=-1,
        )
        q1 = q1.reshape(batch_size, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        k1 = k1.reshape(batch_size, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        v1 = v1.reshape(batch_size, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        q2 = q2.reshape(batch_size, height, width, head_dim).permute(0, 3, 1, 2)
        k2 = k2.reshape(batch_size, height, width, head_dim).permute(0, 3, 1, 2)
        v2 = v2.reshape(batch_size, height, width, head_dim).permute(0, 3, 1, 2)

        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3)
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x2 = F.conv2d(
            q2.reshape(1, batch_size * head_dim, height, width),
            w3,
            padding=1,
            groups=batch_size * head_dim,
        )
        x2 = x2.reshape(batch_size, head_dim, num_tokens).transpose(1, 2)
        return self.proj(torch.cat([x1, x2], dim=-1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class ViTTTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = ViTTTMixer(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = x.shape
        spatial = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = x + self.cpe(spatial).flatten(2).transpose(1, 2)
        x = x + self.mixer(self.norm1(x), height, width)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 96,
        patch_size: int = 5,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class ViTTT2D(nn.Module):
    """ViTTT visual backbone whose public forward accepts raw SQ signals."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        name: str = "vittt2d",
        transform_in_model: bool = True,
        zero_mean: bool = True,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        n_fft: int = 128,
        hop_length: int = 32,
        win_length: int = 128,
        normalize_spectrogram: bool = True,
        patch_size: int = 5,
        embed_dim: int = 96,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.input_channels = input_channels
        self.transform_in_model = transform_in_model
        self.zero_mean = zero_mean
        self.clamp_range = clamp_range
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.normalize_spectrogram = normalize_spectrogram
        self.spectrogram_size = (self.n_fft // 2 + 1, 65)
        self.patch_size = int(patch_size)
        if self.spectrogram_size[0] % self.patch_size or self.spectrogram_size[1] % self.patch_size:
            raise ValueError(f"spectrogram_size={self.spectrogram_size} must be divisible by patch_size={patch_size}")
        self.register_buffer("stft_window", torch.hann_window(self.win_length), persistent=False)

        self.patch_embed = PatchEmbed2D(input_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ViTTTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.feature_dim = int(embed_dim)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.weights_init()

    def weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def normalize_signal(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_in_model:
            x = transform_value(x, zero_mean=self.zero_mean)
        else:
            x = x - x.mean(dim=-1, keepdim=True)
        return torch.clamp(x, *self.clamp_range)

    def signal_to_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected raw signal shape (N, C, L), got {tuple(x.shape)}")

        x = self.normalize_signal(x)
        if x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        signal = x[:, 0, :]
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
        mag = torch.log1p(spec.abs()).unsqueeze(1)
        if self.normalize_spectrogram:
            mean = mag.mean(dim=(-2, -1), keepdim=True)
            std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            mag = (mag - mean) / std
        return torch.clamp(mag, -5.0, 5.0)

    def features_from_image(self, x: torch.Tensor) -> torch.Tensor:
        x, height, width = self.patch_embed(x)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, height, width)
        return self.norm(x).mean(dim=1)

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features_from_image(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_image(self.signal_to_spectrogram(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)


def vittt2d(num_classes=10, input_channels=1, name="vittt2d", **kwargs):
    return ViTTT2D(
        num_classes=num_classes,
        input_channels=input_channels,
        name=name,
        embed_dim=96,
        depth=6,
        num_heads=4,
        mlp_ratio=3.0,
        **kwargs,
    )


def vittt2d_tiny(num_classes=10, input_channels=1, name="vittt2d_tiny", **kwargs):
    return vittt2d(num_classes=num_classes, input_channels=input_channels, name=name, **kwargs)
