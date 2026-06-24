import argparse
import sys
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
import tsaug

try:
    from .common import (
        DataConfig,
        LoaderConfig,
        ModelConfig,
        create_sq_dataloaders,
        build_model,
        TTABase,
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
        get_default_device,
        parse_speeds_arg,
        evaluate_classification,
    )


def get_tta_transforms_1d(gaussian_std: float = 0.02):
    """
    CoTTA 官方 `get_tta_transforms()` 是为 CIFAR 图像设计的。
    这里做严格对齐逻辑的“一维信号版本替换”：提供一个可调用 transform(x)。

    x: torch.Tensor, shape (N, 1, L)
    return: torch.Tensor, shape (N, 1, L)
    """

    def _transform(x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()  # (N, 1, L)
        x_np = x_np.transpose(0, 2, 1)  # (N, L, 1) for tsaug

        # 强增强（对应官方 ColorJitter/Affine/Blur/Noise/Flip 等）
        x_np = tsaug.Reverse(prob=0.5).augment(x_np)
        x_np = tsaug.AddNoise(scale=gaussian_std, prob=1.0).augment(x_np)
        x_np = tsaug.Dropout(p=0.1, fill=0.0, prob=0.5).augment(x_np)

        x_np = x_np.transpose(0, 2, 1)  # (N, 1, L)
        return torch.from_numpy(x_np).to(x.device).float()

    return _transform


def update_ema_variables(ema_model: nn.Module, model: nn.Module, alpha_teacher: float) -> nn.Module:
    # 严格对齐官方：仅对 parameters 做 EMA（官方假设 BN running stats 被禁用）
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1.0 - alpha_teacher) * param[:].data[:]
    return ema_model


@torch.jit.script
def softmax_entropy(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    严格对齐官方 `cifar/cotta.py`：
    loss = - sum( softmax(teacher) * logsoftmax(student) )
    """
    return -(teacher_logits.softmax(1) * student_logits.log_softmax(1)).sum(1)


def copy_model_and_optimizer(model: nn.Module, optimizer: optim.Optimizer):
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model: nn.Module, optimizer: optim.Optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model: nn.Module) -> nn.Module:
    """
    严格对齐官方 `configure_model()`（注意官方 CoTTA 这里把所有模块设为可训练，但 BN 强制 batch stats）。
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
        else:
            m.requires_grad_(True)
    return model


def collect_params(model: nn.Module):
    """
    严格对齐官方 `collect_params()`：收集所有 requires_grad 的 weight/bias。
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters(recurse=False):
            if np in ["weight", "bias"] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


class CoTTAOfficial(TTABase):
    """
    严格对齐官方实现（qinenergy/cotta, cifar/cotta.py），仅将图像增强替换为一维信号增强。
    官方入口逻辑（anchor+ap 门控、N=32 增强平均、soft-label CE、EMA teacher、stochastic restore、episodic reset）保持一致。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        steps: int = 1,
        episodic: bool = False,
        mt_alpha: float = 0.99,
        rst_m: float = 0.1,
        ap: float = 0.9,
        N: int = 32,
        gaussian_std: float = 0.02,
    ):
        super().__init__(model, device=device)
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = copy_model_and_optimizer(
            self.model, self.optimizer
        )
        self.transform = get_tta_transforms_1d(gaussian_std=gaussian_std)

        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.N = N

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)

        # Teacher Prediction: anchor gate
        anchor_prob = torch.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)

        # Augmentation-averaged Prediction
        outputs_emas = []
        for _ in range(self.N):
            out_aug = self.model_ema(self.transform(x)).detach()
            outputs_emas.append(out_aug)

        anchor_mean = float(anchor_prob.mean(0).item())
        use_augavg = anchor_mean < float(self.ap)

        if use_augavg:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        # Student update (soft-label CE)
        loss = softmax_entropy(outputs, outputs_ema).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Teacher update (EMA)
        self.model_ema = update_ema_variables(self.model_ema, self.model, alpha_teacher=self.mt)

        # Stochastic restore (weight/bias)
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters(recurse=False):
                if npp in ["weight", "bias"] and p.requires_grad:
                    mask = (torch.rand(p.shape, device=p.device) < self.rst).float()
                    with torch.no_grad():
                        key = f"{nm}.{npp}" if nm else npp
                        p.data = self.model_state[key].to(p.device) * mask + p * (1.0 - mask)

        return outputs_ema

    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(inputs)
        return outputs.detach()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQ 数据集 CoTTA (Official-aligned) 1D version")
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
    p.add_argument("--device", type=str, default="")

    # CoTTA official hparams
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--episodic", action="store_true")
    p.add_argument("--mt_alpha", type=float, default=0.99)
    p.add_argument("--rst_m", type=float, default=0.1)
    p.add_argument("--ap", type=float, default=0.8)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--gaussian_std", type=float, default=0.001)

    p.add_argument("--no_transform", action="store_true")
    p.add_argument(
        "--corruption_type",
        type=str,
        default=None,
        choices=["noise", "missing"],
        help="数据 Corruption 类型",
    )
    p.add_argument(
        "--severity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Corruption 强度 (1-5)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"[INFO] 使用设备: {device}")

    # Data
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

    # Baseline
    print("\n=== Baseline (No Adapt) ===")
    model_base = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    criterion = nn.CrossEntropyLoss()
    base_metrics = evaluate_classification(model_base, loaders["test"], device=device, criterion=criterion)
    print(f"Baseline Acc: {base_metrics['acc']:.4%}")

    # CoTTA (official-aligned)
    print(
        f"\n=== CoTTA Official-aligned (lr={args.lr}, steps={args.steps}, mt={args.mt_alpha}, rst={args.rst_m}, ap={args.ap}, N={args.N}) ==="
    )
    model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    
    model = configure_model(model)
    params, _ = collect_params(model)
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    # optimizer = optim.SGD(params, lr=1.0)
    
    # params = model.parameters()
    # optimizer = optim.SGD(params, lr=0.1)
    # optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    cotta = CoTTAOfficial(
        model=model,
        optimizer=optimizer,
        device=device,
        steps=args.steps,
        episodic=args.episodic,
        mt_alpha=args.mt_alpha,
        rst_m=args.rst_m,
        ap=args.ap,
        N=args.N,
        gaussian_std=args.gaussian_std,
    )
    tta_metrics = cotta.predict_loader(loaders["test"])
    print(f"CoTTA Acc: {tta_metrics.get('acc', 0.0):.4%}")


if __name__ == "__main__":
    main()


