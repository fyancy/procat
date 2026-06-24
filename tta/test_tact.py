import argparse
import copy
import random
import sys
from typing import Callable, List, Optional, Tuple

import warnings

warnings.filterwarnings("ignore")

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from .common import (
        DataConfig,
        LoaderConfig,
        ModelConfig,
        create_sq_dataloaders,
        build_model,
        TTABase,
        freeze_all,
        get_default_device,
        parse_speeds_arg,
        evaluate_classification,
        enable_singleton_batchnorm_eval,
        softmax_entropy,
    )
    from .test_pasle import extract_features, get_classifier_weight
except ImportError:
    from common import (
        DataConfig,
        LoaderConfig,
        ModelConfig,
        create_sq_dataloaders,
        build_model,
        TTABase,
        freeze_all,
        get_default_device,
        parse_speeds_arg,
        evaluate_classification,
        enable_singleton_batchnorm_eval,
        softmax_entropy,
    )
    from test_pasle import extract_features, get_classifier_weight


def projection(vector_to_project: torch.Tensor, project_direction: torch.Tensor) -> torch.Tensor:
    if vector_to_project.dim() == 3:
        project_direction = project_direction.unsqueeze(-2)
    dot_product = (vector_to_project * project_direction).sum(dim=-1, keepdim=True)
    return dot_product * project_direction


def get_pcs(features: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Eigendecomposition on per-sample augmented feature sets."""
    mean = features.mean(dim=1, keepdim=True)
    x_mean = features - mean
    x_cov = torch.bmm(x_mean.transpose(1, 2), x_mean)
    try:
        _, _, eigenvectors = torch.linalg.svd(x_cov, full_matrices=False)
        return eigenvectors, mean
    except torch.linalg.LinAlgError:
        return None, mean


def remove_pcs(
    features: torch.Tensor,
    pcs: torch.Tensor,
    start_pc: int,
    num_pcs_to_remove: int,
) -> torch.Tensor:
    updated = features
    for i in range(num_pcs_to_remove):
        updated = updated - projection(updated, pcs[:, i + start_pc, :])
    return updated


class SignalNonCausalAugmentation:
    """Non-causal augmentation for 1D SQ signals."""

    def __init__(
        self,
        noise_std: float = 0.05,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        shift_ratio: float = 0.02,
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_ratio = shift_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        scale = random.uniform(*self.scale_range)
        out = out * scale
        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std
        if self.shift_ratio > 0:
            max_shift = max(int(out.shape[-1] * self.shift_ratio), 1)
            shift = random.randint(-max_shift, max_shift)
            out = torch.roll(out, shifts=shift, dims=-1)
        return out


def extract_features_with_aug(
    model: nn.Module,
    x: torch.Tensor,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    if augment is not None:
        if x.dim() == 3:
            x = torch.stack([augment(sample) for sample in x], dim=0)
        else:
            x = augment(x)
    return extract_features(model, x)


class TACT(TTABase):
    """
    Test-time Adaptation via Causal Trimming (TACT).
    Official repo: https://github.com/NancyQuris/TACT
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        num_aug: int = 32,
        start_pc: int = 0,
        num_pcs: int = 4,
        noise_std: float = 0.05,
        scale_range: Tuple[float, float] = (0.85, 1.15),
    ):
        super().__init__(model, device=device)
        self.model.eval()
        freeze_all(self.model)

        self.num_aug = num_aug
        self.start_pc = start_pc
        self.num_pcs = num_pcs
        self.non_causal_aug = SignalNonCausalAugmentation(
            noise_std=noise_std,
            scale_range=scale_range,
        )

        prototypes = get_classifier_weight(self.model).detach().clone()
        self.prototypes = prototypes
        self.updated_prototypes: Optional[torch.Tensor] = None
        self.num_samples_seen = 0
        self._initial_model_state = copy.deepcopy(self.model.state_dict())

    def reset_for_new_sample(self) -> None:
        self.model.load_state_dict(self._initial_model_state, strict=True)
        self.updated_prototypes = None
        self.num_samples_seen = 0

    def _causal_trimming(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        all_features = [features.detach()]

        for _ in range(self.num_aug):
            aug_x = torch.stack(
                [self.non_causal_aug(sample) for sample in x],
                dim=0,
            )
            aug_features = extract_features(self.model, aug_x)
            all_features.append(aug_features.detach())

        stacked = torch.stack(all_features, dim=0).transpose(0, 1)
        pcs, _ = get_pcs(stacked)
        if pcs is None:
            return features

        trimmed_features = remove_pcs(features, pcs, self.start_pc, self.num_pcs)

        batch_size = features.size(0)
        model_prototype = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        projected_prototype = remove_pcs(model_prototype, pcs, self.start_pc, self.num_pcs)
        averaged_prototype = projected_prototype.mean(dim=0)

        if self.updated_prototypes is None:
            self.updated_prototypes = averaged_prototype
        else:
            total = self.num_samples_seen + batch_size
            self.updated_prototypes = (
                self.updated_prototypes * self.num_samples_seen + averaged_prototype * batch_size
            ) / total
        self.num_samples_seen += batch_size

        return trimmed_features

    def _predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        prototypes = self.updated_prototypes if self.updated_prototypes is not None else self.prototypes
        return features @ prototypes.T

    @torch.no_grad()
    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        features = extract_features(self.model, inputs)
        trimmed = self._causal_trimming(inputs, features)
        return self._predict_from_features(trimmed)


class TACTAdapt(TACT):
    """TACT + featurizer adaptation (TACT_adapt in official code)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        adaptation_lr: float = 1e-4,
        entropy_weighting: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, device=device, num_classes=num_classes, **kwargs)
        self.adapt_model = copy.deepcopy(model)
        enable_singleton_batchnorm_eval(self.adapt_model)
        self.adapt_model.train()
        self._set_featurizer_trainable(self.adapt_model)
        self.optimizer = optim.Adam(
            [p for p in self.adapt_model.parameters() if p.requires_grad],
            lr=adaptation_lr,
        )
        self.entropy_weighting = entropy_weighting
        self.criterion = nn.CrossEntropyLoss()
        self._initial_adapt_model_state = copy.deepcopy(self.adapt_model.state_dict())
        self._initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def reset_for_new_sample(self) -> None:
        super().reset_for_new_sample()
        self.adapt_model.load_state_dict(self._initial_adapt_model_state, strict=True)
        self.optimizer.load_state_dict(self._initial_optimizer_state)

    @staticmethod
    def _set_featurizer_trainable(model: nn.Module) -> None:
        freeze_all(model)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            for p in model.parameters():
                p.requires_grad = True
            model.fc.weight.requires_grad = False
            if model.fc.bias is not None:
                model.fc.bias.requires_grad = False
        elif hasattr(model, "classifier"):
            for p in model.parameters():
                p.requires_grad = True
            model.classifier.weight.requires_grad = False
            if model.classifier.bias is not None:
                model.classifier.bias.requires_grad = False

    @torch.enable_grad()
    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        tact_features = extract_features(self.model, inputs)
        trimmed = self._causal_trimming(inputs, tact_features)
        tact_logits = self._predict_from_features(trimmed).detach()

        self.optimizer.zero_grad(set_to_none=True)
        output = self.adapt_model(inputs)
        hard_loss = self.criterion(output, tact_logits.argmax(dim=1))
        entropy = softmax_entropy(output).mean()
        softmax_out = F.softmax(output, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        diversity = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss = hard_loss + self.entropy_weighting * (entropy + diversity)
        loss.backward()
        self.optimizer.step()
        return output.detach()


def build_tact_method(
    model: nn.Module,
    device: torch.device,
    num_classes: int,
    use_adapt: bool,
    num_aug: int,
    start_pc: int,
    num_pcs: int,
    adaptation_lr: float,
    entropy_weighting: float,
    noise_std: float,
) -> TACT:
    common = dict(
        model=model,
        device=device,
        num_classes=num_classes,
        num_aug=num_aug,
        start_pc=start_pc,
        num_pcs=num_pcs,
        noise_std=noise_std,
    )
    if use_adapt:
        return TACTAdapt(
            adaptation_lr=adaptation_lr,
            entropy_weighting=entropy_weighting,
            **common,
        )
    return TACT(**common)


def run_tact_evaluation(
    loaders,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
    num_classes: int,
    use_adapt: bool = True,
    num_aug: int = 8,
    start_pc: int = 0,
    num_pcs: int = 1,
    adaptation_lr: float = 1e-4,
    entropy_weighting: float = 10.0,
    noise_std: float = 0.05,
    reset_each_sample: bool = False,
) -> float:
    """Run one TACT evaluation and return test accuracy."""
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    tact = build_tact_method(
        model=model,
        device=device,
        num_classes=num_classes,
        use_adapt=use_adapt,
        num_aug=num_aug,
        start_pc=start_pc,
        num_pcs=num_pcs,
        adaptation_lr=adaptation_lr,
        entropy_weighting=entropy_weighting,
        noise_std=noise_std,
    )
    metrics = tact.predict_loader(loaders["test"], reset_each_sample=reset_each_sample)
    return float(metrics.get("acc", 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQ 数据集 TACT Test-Time Adaptation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["wdcnn", "resnet18", "resnet34", "resnet50", "resnet101"],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--test_speeds", type=str, default="2,3")
    parser.add_argument("--num_aug", type=int, default=8)
    parser.add_argument("--start_pc", type=int, default=0)
    parser.add_argument("--num_pcs", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--use_adapt", action="store_true", help="Use TACT_adapt variant")
    parser.add_argument("--adaptation_lr", type=float, default=1e-4)
    parser.add_argument("--entropy_weighting", type=float, default=10.0)
    parser.add_argument("--no_transform", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--corruption_type", type=str, default=None, choices=["noise", "missing"])
    parser.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"[INFO] Device: {device}")

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
    model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)

    criterion = nn.CrossEntropyLoss()
    print("\n=== Baseline (No Adapt) ===")
    baseline_metrics = evaluate_classification(
        model, loaders["test"], device=device, criterion=criterion
    )
    print(f"Baseline Acc: {baseline_metrics['acc']:.4%}")

    tact_model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    variant = "TACT_adapt" if args.use_adapt else "TACT"
    tact = build_tact_method(
        model=tact_model,
        device=device,
        num_classes=num_classes,
        use_adapt=args.use_adapt,
        num_aug=args.num_aug,
        start_pc=args.start_pc,
        num_pcs=args.num_pcs,
        adaptation_lr=args.adaptation_lr,
        entropy_weighting=args.entropy_weighting,
        noise_std=args.noise_std,
    )
    print(
        f"\n=== {variant} (num_aug={args.num_aug}, num_pcs={args.num_pcs}, "
        f"start_pc={args.start_pc}, noise_std={args.noise_std}) ==="
    )
    metrics = tact.predict_loader(loaders["test"])
    print(f"{variant} Acc: {metrics.get('acc', 0.0):.4%}")


if __name__ == "__main__":
    main()
