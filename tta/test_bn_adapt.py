import sys
sys.path.append('..')

import argparse
import sys
sys.path.append('..')
from copy import deepcopy

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

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


class BNAdapt(TTABase):
    """
    简单的 BatchNorm 自适应方法：
        - 在推理过程中开启 model.train()，使 BN 统计量根据目标域 batch 更新
    """

    def __init__(self, model: nn.Module, device: torch.device, momentum: float = 0.1):
        super().__init__(model, device=device)
        self.configure_bn_momentum(momentum)
        self._initial_model_state = deepcopy(self.model.state_dict())

    def configure_bn_momentum(self, momentum: float):
        self.model.eval()
        # 调整所有 BN 层的 momentum
        # momentum=1.0 意味着 running stats 完全被当前 batch 替代
        # momentum=0.1 是 PyTorch 默认值
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
                m.momentum = momentum

    def reset_for_new_sample(self) -> None:
        self.model.load_state_dict(self._initial_model_state, strict=True)
        self.configure_bn_momentum(next(
            (
                m.momentum
                for m in self.model.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            ),
            0.1,
        ))

    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        # 仅打开 BN 的统计量更新，不做梯度优化
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
        with torch.no_grad():
            logits = self.model(inputs)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQ 数据集上基于 BatchNorm 统计的简易 Test-Time Adaptation",
    )
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
    parser.add_argument(
        "--test_speeds",
        type=str,
        default="0,1",
        help="目标域测试工况 speeds，例如 '0,1' 或 '2,3' 或 '0,1,2,3'",
    )
    parser.add_argument(
        "--no_transform",
        action="store_true",
        help="关闭 SQDataset 中的归一化变换（transform=False）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="设备，'cpu' 或 'cuda'，为空则自动检测",
    )
    parser.add_argument(
        "--bn_momentum",
        type=float,
        default=0.5,
        help="TTA 时 BN 层的 momentum 值 (默认 0.1, 设为 1.0 可快速适应)",
    )
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
        default=5,
        choices=[0, 1, 2, 3, 4, 5],
        help="Corruption 强度 (1-5)",
    )
    return parser.parse_args()


def run_tta_experiment(
    momentum: float,
    data_cfg: DataConfig,
    loader_cfg: LoaderConfig,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
) -> float:
    # 每次实验重新构建数据和模型，确保隔离
    loaders, _ = create_sq_dataloaders(data_cfg, loader_cfg)
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    
    tta_method = BNAdapt(model, device=device, momentum=momentum)
    print(f"\n[EXPERIMENT] Running BNAdapt with momentum={momentum}...")
    metrics = tta_method.predict_loader(loaders["test"])
    acc = metrics.get('acc', 0.0)
    print(f"  -> Acc: {acc:.4%}")
    return acc


def main():
    args = parse_args()

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_default_device()
    print(f"[INFO] 使用设备: {device}")

    # 数据（使用 TTA 划分：train 只含 9/19Hz，test 含 9/19/29/39Hz）
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
    
    # 第一次获取 num_classes (只需要一次)
    _, num_classes = create_sq_dataloaders(data_cfg, loader_cfg)

    model_cfg = ModelConfig(
        input_length=2048,
        num_classes=num_classes,
        transform_in_model=not args.no_transform,
        zero_mean=True,
        in_channels=1,
        checkpoint_path=args.model_path.strip() or f"checkpoints/{args.model}_sq.pth",
    )

    # 1. Baseline (无 TTA)
    print("\n=== Baseline Evaluation ===")
    base_loaders, _ = create_sq_dataloaders(data_cfg, loader_cfg)
    base_model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    criterion = nn.CrossEntropyLoss()
    base_metrics = evaluate_classification(base_model, base_loaders["test"], device=device, criterion=criterion)
    print(f"Baseline Test Acc: {base_metrics['acc']:.4%}")

    # 2. Compare Momentums
    print("\n=== Comparing BN Momentum ===")
    momentums = [0.01, 0.1, 0.5, 1.0]
    results = {}
    
    for m in momentums:
        acc = run_tta_experiment(m, data_cfg, loader_cfg, model_cfg, args.model, device)
        results[m] = acc

    print("\n=== Summary ===")
    print(f"Baseline: {base_metrics['acc']:.4%}")
    for m, acc in results.items():
        print(f"Momentum {m}: {acc:.4%}")
    
    if len(momentums) > 1:
        best_m = max(results, key=results.get)
        print(f"\nBest Momentum: {best_m} (Acc: {results[best_m]:.4%})")
  

if __name__ == "__main__":
    main()


