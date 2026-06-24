import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Optional
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.sq import get_sq_data, SQDataset
from networks.wdcnn import WDCNN
from networks.tfn import TFN_STTF
from networks.resnet import resnet18, resnet34, resnet50, resnet101
from networks.resnet2d import resnet2d18, resnet2d34, resnet2d50
from networks.ssfn import ssfn
from networks.vittt2d import vittt2d


DeviceType = Literal["cpu", "cuda"]
TargetSplitType = Literal["attack", "test"]


def get_default_device() -> torch.device:
    """获取默认设备，优先使用 GPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_speeds_arg(value: str) -> Tuple[int, ...]:
    """
    解析命令行的 speeds 参数。

    支持格式：
        - "0,1"
        - "2 3"
        - "0, 1, 2, 3"
    """
    s = (value or "").strip()
    if not s:
        return tuple()
    # 同时兼容逗号与空格分隔
    s = s.replace(",", " ")
    parts = [p for p in s.split(" ") if p.strip()]
    return tuple(int(p) for p in parts)


@dataclass
class DataConfig:
    """数据相关配置。"""
    train_ratio: float = 0.8
    cross_domain: bool = False
    transform: bool = True
    augment_train: bool = False
    in_channels: int = 1
    # TTA 相关工况配置
    train_speeds: Tuple[int, ...] = (0, 1)
    test_speeds: Tuple[int, ...] = (0, 1, 2, 3)
    # Corruption 配置
    corruption_type: Optional[str] = None  # 'noise', 'missing', 'noise_dyn'
    severity: int = 0  # 1-5


@dataclass
class LoaderConfig:
    """DataLoader 配置。"""
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    shuffle_train: bool = True
    shuffle_test: bool = True


@dataclass
class ModelConfig:
    """模型相关配置。"""
    input_length: int = 2048
    num_classes: Optional[int] = None  # 不指定时自动从标签推断
    transform_in_model: bool = False
    zero_mean: bool = True
    in_channels: int = 1
    clamp_min: float = -1.0
    clamp_max: float = 1.0
    checkpoint_path: Optional[str] = None
    allow_random_init: bool = False
    ssfn_spectrogram_kind: str = "stft"
    ssfn_spec_erasing: bool = False
    ssfn_base_channels: int = 20


def create_sq_datasets(
    data_cfg: DataConfig,
) -> Tuple[SQDataset, SQDataset, int]:
    """
    基于 SQ 数据集构建训练和测试数据集。
    在 TTA 模式下，测试集包含了所有目标域数据（旧工况+新工况）。

    返回:
        train_dataset: 训练集
        test_dataset: 测试集（含 TTA 目标域数据）
        num_classes: 类别数
    """
    # 约定：SQGenerator 中 speed 维度顺序为 [09Hz, 19Hz, 29Hz, 39Hz]
    x_train, y_train, x_test, y_test = get_sq_data(
        train_ratio=data_cfg.train_ratio,
        train_speeds=data_cfg.train_speeds,
        test_speeds=data_cfg.test_speeds,
        corruption_type=data_cfg.corruption_type,
        severity=data_cfg.severity,
    )

    # 自动推断类别数
    num_classes = int(np.max(y_train)) + 1

    train_dataset = SQDataset(
        x_train,
        y_train,
        transform=data_cfg.transform,
        augment=data_cfg.augment_train,
        in_channels=data_cfg.in_channels,
    )
    test_dataset = SQDataset(
        x_test,
        y_test,
        transform=data_cfg.transform,
        augment=False,
        in_channels=data_cfg.in_channels,
    )

    return train_dataset, test_dataset, num_classes


def create_sq_dataloaders(
    data_cfg: DataConfig,
    loader_cfg: LoaderConfig,
) -> Tuple[Dict[str, DataLoader], int]:
    """
    构建 SQ 数据集对应的 DataLoader。

    返回:
        loaders: {
            'train': train_loader,
            'test': test_loader,
        }
        num_classes: 类别数
    """
    train_dataset, test_dataset, num_classes = create_sq_datasets(data_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle_train,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=True,  # TTA 关键修正：必须打乱，否则 Batch 内类别单一会导致 BN 统计量严重偏移
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )

    if not loader_cfg.shuffle_test:
        test_loader = DataLoader(
            test_dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=False,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
        )

    loaders = {
        "train": train_loader,
        "test": test_loader,
    }
    return loaders, num_classes


def resolve_existing_path(path: str) -> Optional[str]:
    """Resolve a user path from cwd, repo root, or the tta/ directory."""
    if not path:
        return None

    raw = Path(path)
    if raw.is_absolute():
        candidates = [raw]
    else:
        tta_dir = Path(__file__).resolve().parent
        repo_root = tta_dir.parent
        candidates = [Path.cwd() / raw, repo_root / raw, tta_dir / raw]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def load_checkpoint_state_dict(model_cfg: ModelConfig, device: torch.device):
    """Load a checkpoint state dict, failing fast when an expected file is missing."""
    path = model_cfg.checkpoint_path
    if path is None:
        return None

    resolved = resolve_existing_path(path)
    if resolved is None:
        message = f"Checkpoint not found: {path}"
        if model_cfg.allow_random_init:
            print(f"[WARN] {message}; using random initialization.")
            return None
        raise FileNotFoundError(message)

    ckpt = torch.load(resolved, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    print(f"[INFO] Loaded checkpoint: {resolved}")
    return state_dict


def build_wdcnn_model(
    model_cfg: ModelConfig,
    device: Optional[torch.device] = None,
    track_running_stats: bool = True,
) -> nn.Module:
    """
    构建并（可选）加载预训练权重的 WDCNN 模型。
    """
    if device is None:
        device = get_default_device()

    model = WDCNN(
        input_length=model_cfg.input_length,
        num_classes=model_cfg.num_classes or 7,  # 默认 7 类，避免 num_classes 未显式设置时报错
        transform_in_model=model_cfg.transform_in_model,
        zero_mean=model_cfg.zero_mean,
        in_channels=model_cfg.in_channels,
        clamp_range=(model_cfg.clamp_min, model_cfg.clamp_max),
    )

    if model_cfg.checkpoint_path is not None:
        resolved_checkpoint_path = resolve_existing_path(model_cfg.checkpoint_path)
        if resolved_checkpoint_path is None:
            if model_cfg.allow_random_init:
                print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")
                model_cfg.checkpoint_path = None
            else:
                raise FileNotFoundError(f"Checkpoint not found: {model_cfg.checkpoint_path}")
        else:
            model_cfg.checkpoint_path = resolved_checkpoint_path

    if model_cfg.checkpoint_path is not None and os.path.isfile(model_cfg.checkpoint_path):
        ckpt = torch.load(model_cfg.checkpoint_path, map_location=device)
        # 兼容 {'model': state_dict} 或直接 state_dict
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] 加载预训练模型: {model_cfg.checkpoint_path}")
    elif model_cfg.checkpoint_path:
        print(f"[WARN] 未找到 checkpoint 文件: {model_cfg.checkpoint_path}，将使用随机初始化权重。")

    model.to(device)
    # todo: set track_running_stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = track_running_stats
    
    return model


def build_tfn_model(
    model_cfg: ModelConfig,
    device: Optional[torch.device] = None,
    track_running_stats: bool = True,
) -> nn.Module:
    """Build TFN_STTF and optionally load a checkpoint."""
    if device is None:
        device = get_default_device()

    model = TFN_STTF(
        input_length=model_cfg.input_length,
        num_classes=model_cfg.num_classes or 7,
        transform_in_model=model_cfg.transform_in_model,
        zero_mean=model_cfg.zero_mean,
        in_channels=model_cfg.in_channels,
        clamp_range=(model_cfg.clamp_min, model_cfg.clamp_max),
        track_running_stats=track_running_stats,
    )

    if model_cfg.checkpoint_path is not None:
        resolved_checkpoint_path = resolve_existing_path(model_cfg.checkpoint_path)
        if resolved_checkpoint_path is None:
            if model_cfg.allow_random_init:
                print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")
                model_cfg.checkpoint_path = None
            else:
                raise FileNotFoundError(f"Checkpoint not found: {model_cfg.checkpoint_path}")
        else:
            model_cfg.checkpoint_path = resolved_checkpoint_path

    if model_cfg.checkpoint_path is not None and os.path.isfile(model_cfg.checkpoint_path):
        ckpt = torch.load(model_cfg.checkpoint_path, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] 加载预训练模型: {model_cfg.checkpoint_path}")
    elif model_cfg.checkpoint_path:
        print(f"[WARN] 未找到 checkpoint 文件: {model_cfg.checkpoint_path}，将使用随机初始化权重。")

    model.to(device)
    model._set_bn_track_running_stats(track_running_stats)
    return model


def build_resnet_model(
    model_cfg: ModelConfig,
    device: Optional[torch.device] = None,
    track_running_stats: bool = True,
    arch: str = "resnet18",
) -> nn.Module:
    """
    构建并（可选）加载预训练权重的 ResNet1D 模型。

    arch:
        - 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101'
    """
    if device is None:
        device = get_default_device()

    builders = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
    }
    if arch not in builders:
        raise ValueError(f"Unknown ResNet arch: {arch}")

    model = builders[arch](
        num_classes=model_cfg.num_classes or 7,
        input_channels=model_cfg.in_channels,
        name=arch,
        transform_in_model=model_cfg.transform_in_model,
        zero_mean=model_cfg.zero_mean,
    )

    if model_cfg.checkpoint_path is not None:
        resolved_checkpoint_path = resolve_existing_path(model_cfg.checkpoint_path)
        if resolved_checkpoint_path is None:
            if model_cfg.allow_random_init:
                print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")
                model_cfg.checkpoint_path = None
            else:
                raise FileNotFoundError(f"Checkpoint not found: {model_cfg.checkpoint_path}")
        else:
            model_cfg.checkpoint_path = resolved_checkpoint_path

    if model_cfg.checkpoint_path is not None and os.path.isfile(model_cfg.checkpoint_path):
        ckpt = torch.load(model_cfg.checkpoint_path, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] 加载预训练模型: {model_cfg.checkpoint_path}")
    elif model_cfg.checkpoint_path:
        print(f"[WARN] 未找到 checkpoint 文件: {model_cfg.checkpoint_path}，将使用随机初始化权重。")

    model.to(device)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = track_running_stats

    return model


def build_resnet2d_model(
    model_cfg: ModelConfig,
    device: Optional[torch.device] = None,
    track_running_stats: bool = True,
    arch: str = "resnet2d18",
) -> nn.Module:
    """Build and optionally load a 2D STFT-spectrogram ResNet."""
    del track_running_stats
    if device is None:
        device = get_default_device()

    builders = {
        "resnet2d18": resnet2d18,
        "resnet2d34": resnet2d34,
        "resnet2d50": resnet2d50,
    }
    if arch not in builders:
        raise ValueError(f"Unknown ResNet2D arch: {arch}")

    model = builders[arch](
        num_classes=model_cfg.num_classes or 7,
        input_channels=model_cfg.in_channels,
        name=arch,
        transform_in_model=model_cfg.transform_in_model,
        zero_mean=model_cfg.zero_mean,
        clamp_range=(model_cfg.clamp_min, model_cfg.clamp_max),
    )

    if model_cfg.checkpoint_path is not None:
        resolved_checkpoint_path = resolve_existing_path(model_cfg.checkpoint_path)
        if resolved_checkpoint_path is None:
            if model_cfg.allow_random_init:
                print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")
                model_cfg.checkpoint_path = None
            else:
                raise FileNotFoundError(f"Checkpoint not found: {model_cfg.checkpoint_path}")
        else:
            model_cfg.checkpoint_path = resolved_checkpoint_path

    if model_cfg.checkpoint_path is not None and os.path.isfile(model_cfg.checkpoint_path):
        ckpt = torch.load(model_cfg.checkpoint_path, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded ResNet2D checkpoint: {model_cfg.checkpoint_path}")
    elif model_cfg.checkpoint_path:
        print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")

    model.to(device)
    return model


def build_vittt2d_model(
    model_cfg: ModelConfig,
    device: Optional[torch.device] = None,
    track_running_stats: bool = True,
) -> nn.Module:
    """Build and optionally load a ViTTT-style 2D spectrogram backbone."""
    del track_running_stats
    if device is None:
        device = get_default_device()

    model = vittt2d(
        num_classes=model_cfg.num_classes or 7,
        input_channels=model_cfg.in_channels,
        name="vittt2d",
        transform_in_model=model_cfg.transform_in_model,
        zero_mean=model_cfg.zero_mean,
        clamp_range=(model_cfg.clamp_min, model_cfg.clamp_max),
    )

    if model_cfg.checkpoint_path is not None:
        resolved_checkpoint_path = resolve_existing_path(model_cfg.checkpoint_path)
        if resolved_checkpoint_path is None:
            if model_cfg.allow_random_init:
                print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")
                model_cfg.checkpoint_path = None
            else:
                raise FileNotFoundError(f"Checkpoint not found: {model_cfg.checkpoint_path}")
        else:
            model_cfg.checkpoint_path = resolved_checkpoint_path

    if model_cfg.checkpoint_path is not None and os.path.isfile(model_cfg.checkpoint_path):
        ckpt = torch.load(model_cfg.checkpoint_path, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded ViTTT2D checkpoint: {model_cfg.checkpoint_path}")
    elif model_cfg.checkpoint_path:
        print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")

    model.to(device)
    return model


def build_ssfn_model(
    model_cfg: ModelConfig,
    device: Optional[torch.device] = None,
    track_running_stats: bool = True,
) -> nn.Module:
    """Build and optionally load the sequence-spectrogram fusion network."""
    if device is None:
        device = get_default_device()

    model = ssfn(
        input_length=model_cfg.input_length,
        num_classes=model_cfg.num_classes or 7,
        input_channels=model_cfg.in_channels,
        name="ssfn",
        transform_in_model=model_cfg.transform_in_model,
        zero_mean=model_cfg.zero_mean,
        clamp_range=(model_cfg.clamp_min, model_cfg.clamp_max),
        spectrogram_kind=model_cfg.ssfn_spectrogram_kind,
        spec_erasing=model_cfg.ssfn_spec_erasing,
        base_channels=model_cfg.ssfn_base_channels,
    )

    if model_cfg.checkpoint_path is not None:
        resolved_checkpoint_path = resolve_existing_path(model_cfg.checkpoint_path)
        if resolved_checkpoint_path is None:
            if model_cfg.allow_random_init:
                print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")
                model_cfg.checkpoint_path = None
            else:
                raise FileNotFoundError(f"Checkpoint not found: {model_cfg.checkpoint_path}")
        else:
            model_cfg.checkpoint_path = resolved_checkpoint_path

    if model_cfg.checkpoint_path is not None and os.path.isfile(model_cfg.checkpoint_path):
        ckpt = torch.load(model_cfg.checkpoint_path, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded SSFN checkpoint: {model_cfg.checkpoint_path}")
    elif model_cfg.checkpoint_path:
        print(f"[WARN] Checkpoint not found: {model_cfg.checkpoint_path}; using random initialization.")

    model.to(device)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = track_running_stats
    return model


def build_model(
    model_cfg: ModelConfig,
    model_name: str = "wdcnn",
    device: Optional[torch.device] = None,
    track_running_stats: bool = True,
) -> nn.Module:
    """
    统一模型构建入口：训练/测试时通过 model_name 选择不同 backbone。

    model_name:
        - 'wdcnn'
        - 'tfn' | 'tfn_sttf'
        - 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101'
    """
    if model_name in {"tfn", "tfn_sttf"}:
        return build_tfn_model(model_cfg, device=device, track_running_stats=track_running_stats)
    if model_name == "wdcnn":
        return build_wdcnn_model(model_cfg, device=device, track_running_stats=track_running_stats)
    if model_name == "ssfn":
        return build_ssfn_model(model_cfg, device=device, track_running_stats=track_running_stats)
    if model_name in {"vittt2d", "vittt2d_tiny"}:
        return build_vittt2d_model(model_cfg, device=device, track_running_stats=track_running_stats)
    if model_name.startswith("resnet"):
        if model_name.startswith("resnet2d"):
            return build_resnet2d_model(
                model_cfg,
                device=device,
                track_running_stats=track_running_stats,
                arch=model_name,
            )
        return build_resnet_model(
            model_cfg,
            device=device,
            track_running_stats=track_running_stats,
            arch=model_name,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_classification(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    标准分类评估（无 TTA 适配）。
    """
    if device is None:
        device = get_default_device()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = torch.from_numpy(inputs) if isinstance(inputs, np.ndarray) else inputs
            targets = torch.from_numpy(targets) if isinstance(targets, np.ndarray) else targets

            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).long()

            logits = model(inputs)
            loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            correct = (preds == targets).sum().item()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size

            if max_batches is not None and (i + 1) >= max_batches:
                break

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return {"loss": avg_loss, "acc": acc}


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算预测分布的熵，用于 TENT 等方法。
    """
    prob = torch.softmax(logits, dim=1)
    log_prob = torch.log_softmax(logits, dim=1)
    entropy = -(prob * log_prob).sum(dim=1)
    return entropy


def freeze_all(model: nn.Module) -> None:
    """冻结模型所有参数梯度。"""
    for p in model.parameters():
        p.requires_grad = False


def select_bn_params(model: nn.Module):
    """
    选择 BatchNorm 层的仿射参数（weight / bias），
    用于 TENT 中仅更新 BN 参数。
    """
    bn_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                bn_params.append(m.weight)
            if m.bias is not None:
                bn_params.append(m.bias)
    return bn_params


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _values_per_bn_channel(x: torch.Tensor) -> int:
    if x.dim() < 2:
        return int(x.numel())
    channel_dim = max(int(x.shape[1]), 1)
    return int(x.numel() // channel_dim)


def _ensure_batchnorm_eval_stats(module: nn.Module, device=None, dtype=None) -> None:
    if getattr(module, "running_mean", None) is None:
        ref = module.weight if getattr(module, "weight", None) is not None else None
        device = device or (ref.device if ref is not None else None)
        dtype = dtype or (ref.dtype if ref is not None else torch.float32)
        module.running_mean = torch.zeros(module.num_features, device=device, dtype=dtype)
    if getattr(module, "running_var", None) is None:
        ref = module.weight if getattr(module, "weight", None) is not None else None
        device = device or (ref.device if ref is not None else None)
        dtype = dtype or (ref.dtype if ref is not None else torch.float32)
        module.running_var = torch.ones(module.num_features, device=device, dtype=dtype)


def _singleton_batchnorm_pre_hook(module: nn.Module, inputs) -> None:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return
    x = inputs[0]
    if _values_per_bn_channel(x) > 1:
        return

    _ensure_batchnorm_eval_stats(module, device=x.device, dtype=x.dtype)
    module.eval()


def enable_singleton_batchnorm_eval(model: nn.Module) -> None:
    """Avoid BatchNorm failures for single-sample 2D activations.

    This only affects BN layers whose current input has one value per channel,
    e.g. WDCNN's final ``BatchNorm1d(100)`` on a batch of one. Convolutional BN
    layers with a time dimension still run in their configured mode.
    """
    already_installed = getattr(model, "_singleton_bn_eval_hooks_installed", False)
    for module in model.modules():
        if isinstance(module, BN_TYPES):
            _ensure_batchnorm_eval_stats(module)
            if already_installed:
                continue
            module.register_forward_pre_hook(_singleton_batchnorm_pre_hook)
    if already_installed:
        return
    model._singleton_bn_eval_hooks_installed = True


class TTABase(ABC):
    """
    TTA 适配方法基类。

    约定：
        - 所有实现类都不修改外部 dataloader，只对内部模型进行更新
        - 每个 batch 的适配过程由 adapt_one_batch 完成
        - 支持返回预测结果以便统一评估
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or get_default_device()
        self.model.to(self.device)
        enable_singleton_batchnorm_eval(self.model)

    @abstractmethod
    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        对单个 batch 进行一次适配，并返回该 batch 的 logits。
        具体优化逻辑由子类实现。
        """

    def predict_loader(
        self,
        data_loader: DataLoader,
        targets: Optional[torch.Tensor] = None,
        reset_each_sample: bool = False,
    ) -> Dict[str, float]:
        """
        在给定 DataLoader 上执行 TTA 推理与评估。
        如果提供 targets，则同时计算准确率。
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        all_logits = []
        all_labels = []

        for inputs, labels in data_loader:
            inputs = torch.from_numpy(inputs) if isinstance(inputs, np.ndarray) else inputs
            labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels

            inputs = inputs.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True).long()

            if reset_each_sample:
                batch_logits = []
                for sample_idx in range(inputs.size(0)):
                    if hasattr(self, "reset_for_new_sample"):
                        self.reset_for_new_sample()
                    sample_logits = self.adapt_one_batch(inputs[sample_idx : sample_idx + 1])
                    batch_logits.append(sample_logits.detach())
                logits = torch.cat(batch_logits, dim=0)
            else:
                logits = self.adapt_one_batch(inputs)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            batch_size = labels.size(0)

            total_correct += correct
            total_samples += batch_size

        metrics = {}
        if total_samples > 0:
            metrics["acc"] = total_correct / total_samples

        return metrics




