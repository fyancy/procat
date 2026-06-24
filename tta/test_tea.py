import argparse
import sys
from copy import deepcopy

import warnings

warnings.filterwarnings("ignore")

sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim

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
        enable_singleton_batchnorm_eval,
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
        enable_singleton_batchnorm_eval,
    )


def init_random(bs, im_sz=2048, n_ch=1):
    """初始化 replay buffer 用的随机样本，均匀分布在 [-1, 1].

    为严格对齐官方 `energy.py` 的接口，这里保留参数名 `im_sz`；在 1D 信号场景下
    它代表 signal length。
    """
    return torch.FloatTensor(bs, n_ch, im_sz).uniform_(-1, 1)


class EnergyModel(nn.Module):
    """
    将分类模型包装为 energy-based model：
    - classify(x): 返回 logits
    - forward(x): 返回 (logsumexp, logits)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.f = model

    def classify(self, x):
        return self.f(x)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits


def sample_p_0(reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
    """从 replay buffer 或随机初始化中采样初始样本。"""
    if len(replay_buffer) == 0:
        return init_random(bs, im_sz=im_sz, n_ch=n_ch), []

    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))

    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs, im_sz=im_sz, n_ch=n_ch)
    choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(
    f: EnergyModel,
    replay_buffer: torch.Tensor,
    n_steps: int,
    sgld_lr: float,
    sgld_std: float,
    reinit_freq: float,
    batch_size: int,
    im_sz: int,
    n_ch: int,
    device: torch.device,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    y=None,
):
    """
    使用 SGLD 在输入空间中采样 x_fake，用于构造对比能量。
    这是对官方实现中 sample_q 的一维信号版本改写。
    """
    f.eval()
    bs = batch_size if y is None else y.size(0)

    init_sample, buffer_inds = sample_p_0(
        reinit_freq=reinit_freq,
        replay_buffer=replay_buffer,
        bs=bs,
        im_sz=im_sz,
        n_ch=n_ch,
        device=device,
        y=y,
    )
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)

    for _ in range(n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
        x_k.data.clamp_(clip_min, clip_max)

    f.train()
    final_samples = x_k.detach().clamp(clip_min, clip_max)

    if len(replay_buffer) > 0 and isinstance(buffer_inds, torch.Tensor):
        replay_buffer[buffer_inds] = final_samples.cpu()

    return final_samples, init_samples.detach()


@torch.enable_grad()
def forward_and_adapt(
    x: torch.Tensor,
    energy_model: EnergyModel,
    optimizer: optim.Optimizer,
    replay_buffer: torch.Tensor,
    sgld_steps: int,
    sgld_lr: float,
    sgld_std: float,
    reinit_freq: float,
    if_cond=False,
    n_classes: int = 10,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
):
    """
    官方 TEA energy.py 的 forward_and_adapt 的一维信号改写版本：
    - 先用 SGLD 采样 x_fake
    - 计算 real/fake 的能量差
    - 最小化 -(E_real - E_fake)
    """
    batch_size = x.shape[0]
    n_ch = x.shape[1]
    im_sz = x.shape[2]
    device = x.device

    if if_cond == "uncond":
        x_fake, _ = sample_q(
            energy_model,
            replay_buffer,
            n_steps=sgld_steps,
            sgld_lr=sgld_lr,
            sgld_std=sgld_std,
            reinit_freq=reinit_freq,
            batch_size=batch_size,
            im_sz=im_sz,
            n_ch=n_ch,
            device=device,
            clip_min=clip_min,
            clip_max=clip_max,
            y=None,
        )
    elif if_cond == "cond":
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q(
            energy_model,
            replay_buffer,
            n_steps=sgld_steps,
            sgld_lr=sgld_lr,
            sgld_std=sgld_std,
            reinit_freq=reinit_freq,
            batch_size=batch_size,
            im_sz=im_sz,
            n_ch=n_ch,
            device=device,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
        )
    else:
        raise ValueError(f"Unknown if_cond mode: {if_cond}")

    out_real = energy_model(x)
    energy_real = out_real[0].mean()
    energy_fake = energy_model(x_fake)[0].mean()

    loss = -(energy_real - energy_fake)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    outputs = energy_model.classify(x)
    return outputs


def copy_model_and_optimizer(model: nn.Module, optimizer: optim.Optimizer):
    """严格对齐官方：保存 model/optimizer 的状态副本，用于 episodic reset。"""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model: nn.Module, optimizer: optim.Optimizer, model_state, optimizer_state):
    """严格对齐官方：加载 model/optimizer 状态。"""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model: nn.Module, adapt_params: str = "bn") -> nn.Module:
    if adapt_params not in {"bn", "affine", "all"}:
        raise ValueError(f"Unknown adapt_params: {adapt_params}")
    """对齐官方 TENT/TEA：仅 BN 可训练，并强制使用 batch stats。"""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            if adapt_params == "bn":
                m.requires_grad_(True)
        elif adapt_params == "all":
            m.requires_grad_(True)
        if adapt_params == "affine":
            for name, p in m.named_parameters(recurse=False):
                if name in ["weight", "bias"]:
                    p.requires_grad = True
    if adapt_params == "all":
        model.requires_grad_(True)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model


def collect_params(model: nn.Module, adapt_params: str = "bn"):
    if adapt_params not in {"bn", "affine", "all"}:
        raise ValueError(f"Unknown adapt_params: {adapt_params}")
    """对齐官方 collect_params：收集所有 requires_grad 的 weight/bias。"""
    params = []
    names = []
    if adapt_params == "all":
        for name, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(name)
        return params, names
    for nm, m in model.named_modules():
        if adapt_params == "bn" and not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        for np, p in m.named_parameters(recurse=False):
            if np in ["weight", "bias"] and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


class Energy(nn.Module):
    """严格对齐官方 `core/adazoo/energy.py` 的封装（1D 信号版本）。

    官方接口保持一致：
    - `forward(x, if_adapt=True, counter=None, if_vis=False)`
    - 支持 `episodic=True` 时每个 batch 前 reset
    """

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        episodic=False,
        buffer_size=10000,
        sgld_steps=20,
        sgld_lr=1,
        sgld_std=0.01,
        reinit_freq=0.05,
        if_cond=False,
        n_classes=10,
        im_sz=2048,
        n_ch=1,
        clip_min=-1.0,
        clip_max=1.0,
        path=None,
        logger=None,
    ):
        super().__init__()

        self.energy_model = EnergyModel(model)
        self.replay_buffer = init_random(buffer_size, im_sz=im_sz, n_ch=n_ch)
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.if_cond = if_cond

        self.n_classes = n_classes
        self.im_sz = im_sz
        self.n_ch = n_ch
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.path = path
        self.logger = logger

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.energy_model, self.optimizer)

    def forward(self, x, if_adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()

        if if_adapt:
            # self.energy_model(x)
            for i in range(self.steps):
                outputs = forward_and_adapt(
                    x,
                    self.energy_model,
                    self.optimizer,
                    self.replay_buffer,
                    self.sgld_steps,
                    self.sgld_lr,
                    self.sgld_std,
                    self.reinit_freq,
                    if_cond=self.if_cond,
                    n_classes=self.n_classes,
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                )
                # 对齐官方接口：保留可视化形参
                _ = (counter, if_vis, i)  # noqa: F841
        else:
            self.energy_model.eval()
            with torch.no_grad():
                outputs = self.energy_model.classify(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.energy_model, self.optimizer, self.model_state, self.optimizer_state)
        self.replay_buffer = deepcopy(self.replay_buffer_old)


class TEAOfficial(TTABase):
    """TEA Official-aligned（按官方 `Energy` 封装执行）。"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
        buffer_size: int = 10000,
        sgld_steps: int = 20,
        sgld_lr: float = 1.0,
        sgld_std: float = 0.01,
        reinit_freq: float = 0.05,
        if_cond: str = "uncond",
        n_classes: int = 10,
        im_sz: int = 2048,
        n_ch: int = 1,
        adapt_params: str = "bn",
        clip_min: float = -1.0,
        clip_max: float = 1.0,
    ):
        super().__init__(model, device=device)

        self.model = configure_model(self.model, adapt_params=adapt_params)
        enable_singleton_batchnorm_eval(self.model)
        self.energy = Energy(
            model=self.model,
            optimizer=optimizer,
            steps=steps,
            episodic=episodic,
            buffer_size=buffer_size,
            sgld_steps=sgld_steps,
            sgld_lr=sgld_lr,
            sgld_std=sgld_std,
            reinit_freq=reinit_freq,
            if_cond=if_cond,
            n_classes=n_classes,
            im_sz=im_sz,
            n_ch=n_ch,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        # 对齐官方：确保 BN 使用 batch stats
        self.model.train()
        outputs = self.energy(inputs, if_adapt=True)
        return outputs.detach()

    def reset_for_new_sample(self) -> None:
        self.energy.reset()


def run_tea_evaluation(
    loaders,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
    num_classes: int,
    lr: float = 1e-3,
    adapt_params: str = "bn",
    optimizer_name: str = "sgd",
    steps: int = 1,
    sgld_steps: int = 1,
    sgld_lr: float = 1.0,
    sgld_std: float = 0.01,
    reinit_freq: float = 0.05,
    buffer_size: int = 10000,
    sgld_clip_min: float = -1.0,
    sgld_clip_max: float = 1.0,
    reset_each_sample: bool = False,
) -> float:
    """Run one TEA evaluation and return test accuracy."""
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    model = configure_model(model, adapt_params=adapt_params)
    params, _ = collect_params(model, adapt_params=adapt_params)
    if not params:
        raise RuntimeError("TEA: no trainable parameters found.")

    if optimizer_name == "adam":
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(params, lr=lr)

    tea = TEAOfficial(
        model=model,
        device=device,
        optimizer=optimizer,
        steps=steps,
        buffer_size=buffer_size,
        sgld_steps=sgld_steps,
        sgld_lr=sgld_lr,
        sgld_std=sgld_std,
        reinit_freq=reinit_freq,
        n_classes=num_classes,
        adapt_params=adapt_params,
        clip_min=sgld_clip_min,
        clip_max=sgld_clip_max,
    )
    metrics = tea.predict_loader(loaders["test"], reset_each_sample=reset_each_sample)
    return float(metrics.get("acc", 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQ 数据集 TEA (Test-time Energy Adaptation)")
    parser.add_argument("--batch_size", type=int, default=256, help="评估批大小")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集划分比例")
    parser.add_argument(
        "--model",
        type=str,
        default="wdcnn",
        choices=["wdcnn", "resnet18", "resnet34", "resnet50", "resnet101"],
        help="选择 backbone 模型",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="预训练模型 checkpoint 路径（为空则使用 checkpoints/{model}_sq.pth）",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument(
        "--test_speeds",
        type=str,
        default="2,3",
        help="目标域测试工况 speeds，例如 '0,1' 或 '2,3' 或 '0,1,2,3'",
    )
    parser.add_argument("--steps", type=int, default=1, help="Adaptation steps per batch")
    parser.add_argument("--adapt_params", type=str, default="bn", choices=["bn", "affine", "all"])
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--sgld_clip_min", type=float, default=-1.0)
    parser.add_argument("--sgld_clip_max", type=float, default=1.0)
    parser.add_argument("--sgld_steps", type=int, default=20, help="SGLD 迭代步数")
    parser.add_argument("--sgld_lr", type=float, default=1.0, help="SGLD 学习率")
    parser.add_argument("--sgld_std", type=float, default=0.01, help="SGLD 噪声标准差")
    parser.add_argument("--reinit_freq", type=float, default=0.05, help="SGLD 重置频率")
    parser.add_argument("--episodic", action="store_true", help="每个 batch 前重置回源模型状态（官方 episodic）")
    parser.add_argument("--no_transform", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--corruption_type",
        type=str,
        default=None,
        choices=["noise", "missing"],
        help="数据 Corruption 类型",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Corruption 强度 (1-5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sgld_clip_min >= args.sgld_clip_max:
        raise ValueError("--sgld_clip_min must be smaller than --sgld_clip_max")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_default_device()
    print(f"[INFO] Device: {device}")

    # Data: 这里默认做 domain adaptation（train: 0/1Hz, test: 2/3Hz，可根据需要调整）
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

    # Model Configuration
    model_cfg = ModelConfig(
        input_length=2048,
        num_classes=num_classes,
        transform_in_model=not args.no_transform,
        zero_mean=True,
        in_channels=1,
        checkpoint_path=args.model_path.strip() or f"checkpoints/{args.model}_sq.pth",
    )

    # 1. Baseline (No Adapt)
    # Important: Use track_running_stats=True for Baseline to reflect source performance correctly
    print("\n=== Baseline (No Adapt) ===")
    model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    criterion = nn.CrossEntropyLoss()
    base_metrics = evaluate_classification(model, loaders["test"], device=device, criterion=criterion)
    print(f"Baseline Acc: {base_metrics['acc']:.4%}")

    # 2. TEA
    print(
        f"\n=== TEA (Energy-based, lr={args.lr}, steps={args.steps}, "
        f"adapt_params={args.adapt_params}, optimizer={args.optimizer}, "
        f"sgld_clip=[{args.sgld_clip_min}, {args.sgld_clip_max}]) ==="
    )
    # 重新构建模型用于 TEA 适配
    model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    # model = configure_model(model)
    # params, _ = collect_params(model)  # use bn
    model = configure_model(model, adapt_params=args.adapt_params)
    params, names = collect_params(model, adapt_params=args.adapt_params)
    if not params:
        raise RuntimeError("TEAOfficial: 没有找到任何可训练的 weight/bias（请检查 BN 是否存在）。")
    print(f"[INFO] TEA trainable params: {len(params)} tensors ({sum(p.numel() for p in params)} scalars)")
    if args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(params, lr=args.lr)

    tea = TEAOfficial(
        model=model,
        device=device,
        optimizer=optimizer,
        steps=args.steps,
        episodic=args.episodic,
        buffer_size=10000,
        sgld_steps=args.sgld_steps,
        sgld_lr=args.sgld_lr,
        sgld_std=args.sgld_std,
        reinit_freq=args.reinit_freq,
        if_cond="uncond",
        n_classes=num_classes,
        im_sz=2048,
        n_ch=1,
        adapt_params=args.adapt_params,
        clip_min=args.sgld_clip_min,
        clip_max=args.sgld_clip_max,
    )
    tta_metrics = tea.predict_loader(loaders["test"])
    print(f"TEA Acc: {tta_metrics.get('acc', 0.0):.4%}")


if __name__ == "__main__":
    main()
