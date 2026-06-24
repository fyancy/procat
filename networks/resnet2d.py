"""2D ResNet backbones for STFT spectrogram-based SQ experiments."""

import sys
from typing import Tuple

sys.path.append("..")

import torch
import torch.nn as nn

from utils.ts_transform import transform_value


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck2D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet2D(nn.Module):
    """ResNet classifier whose input image is an STFT log-magnitude spectrogram.

    The public ``forward`` accepts raw SQ signals shaped ``(N, 1, L)`` and
    converts them to normalized spectrograms internally.  TTT2D can also call
    ``forward_image`` and ``features_from_image`` directly for SSL views.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        input_channels=1,
        name="resnet2d",
        transform_in_model=True,
        zero_mean=True,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        n_fft: int = 128,
        hop_length: int = 32,
        win_length: int = 128,
        normalize_spectrogram: bool = True,
    ):
        super().__init__()
        self.name = name
        self.in_planes = 64
        self.input_channels = input_channels
        self.transform_in_model = transform_in_model
        self.zero_mean = zero_mean
        self.clamp_range = clamp_range
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.normalize_spectrogram = normalize_spectrogram
        self.spectrogram_size = (self.n_fft // 2 + 1, 65)
        self.register_buffer("stft_window", torch.hann_window(self.win_length), persistent=False)

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512 * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)

        self.weights_init()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def normalize_signal(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_in_model:
            x = transform_value(x, zero_mean=self.zero_mean)
        else:
            x = x - x.mean(dim=-1, keepdim=True)
        x = torch.clamp(x, *self.clamp_range)
        return x

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features_from_image(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_image(self.signal_to_spectrogram(x))

    def predict(self, x: torch.Tensor):
        self.eval()
        return self.forward(x)


def resnet2d18(num_classes=10, input_channels=1, name="resnet2d18", **kwargs):
    return ResNet2D(BasicBlock2D, [2, 2, 2, 2], num_classes, input_channels, name=name, **kwargs)


def resnet2d34(num_classes=10, input_channels=1, name="resnet2d34", **kwargs):
    return ResNet2D(BasicBlock2D, [3, 4, 6, 3], num_classes, input_channels, name=name, **kwargs)


def resnet2d50(num_classes=10, input_channels=1, name="resnet2d50", **kwargs):
    return ResNet2D(Bottleneck2D, [3, 4, 6, 3], num_classes, input_channels, name=name, **kwargs)
