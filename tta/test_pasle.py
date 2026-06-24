import argparse
import sys
from copy import deepcopy
from typing import List, Optional, Tuple

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
        softmax_entropy,
    )
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
        softmax_entropy,
    )


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def cc_loss(outputs: torch.Tensor, partial_y: torch.Tensor, temp: float) -> torch.Tensor:
    """Complementary label loss used in PASLE."""
    sm_outputs = F.softmax(outputs / temp, dim=1)
    final_outputs = sm_outputs * partial_y
    return -torch.log(final_outputs.sum(dim=1).clamp_min(1e-8)).mean()


def collect_affine_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect affine weight/bias parameters (BN + Linear)."""
    params: List[nn.Parameter] = []
    names: List[str] = []
    for nm, m in model.named_modules():
        for np_name, p in m.named_parameters(recurse=False):
            if np_name in ("weight", "bias") and p.requires_grad:
                params.append(p)
                names.append(f"{nm}.{np_name}" if nm else np_name)
    return params, names


def configure_model_for_pasle(model: nn.Module, update_param: str = "all") -> nn.Module:
    """Configure trainable parameters for PASLE (aligned with official update_param)."""
    if update_param not in {"all", "affine"}:
        raise ValueError(f"Unknown update_param: {update_param}")

    model.train()
    if update_param == "all":
        model.requires_grad_(True)
        return model

    freeze_all(model)
    for m in model.modules():
        if isinstance(m, BN_TYPES):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        for name, p in m.named_parameters(recurse=False):
            if name in ("weight", "bias"):
                p.requires_grad = True
    return model


def get_classifier_weight(model: nn.Module) -> torch.Tensor:
    """Return class-prototype matrix [num_classes, feat_dim]."""
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        return model.classifier.weight.data
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.weight.data
    raise RuntimeError("PASLE_E requires a final Linear classifier with accessible weight.")


def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Feature extractor before the final classifier."""
    if hasattr(model, "features") and callable(getattr(model, "features")):
        return model.features(x)

    if hasattr(model, "signal_to_spectrogram") and hasattr(model, "features_from_image"):
        image = model.signal_to_spectrogram(x)
        return model.features_from_image(image)

    if hasattr(model, "conv1"):
        x = model.transform_fn(x)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    x = model.transform_fn(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.layer5(x)
    z = x.view(x.size(0), -1)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        if hasattr(model, "fc") and isinstance(model.fc, nn.Sequential):
            return model.fc(z)
    return z


def classify_features(model: nn.Module, z: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "classify_features") and callable(getattr(model, "classify_features")):
        return model.classify_features(z)

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        if hasattr(model, "fc") and isinstance(model.fc, nn.Sequential):
            if z.shape[-1] == model.classifier.in_features:
                return model.classifier(z)
            return model.classifier(model.fc(z))
        return model.classifier(z)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc(z)
    raise RuntimeError("Unsupported model head for PASLE classifier forward.")


def predict_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.train()
    return model(x)


class PASLE(TTABase):
    """
    Selective Label Enhancement Learning (ICLR 2025).
    Official repo: https://github.com/palm-ml/PASLE
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        thresh: float = 0.8,
        thresh_gap: float = 0.1,
        thresh_des: float = 0.001,
        temp: float = 5.0,
        buffer_size: int = 32,
    ):
        super().__init__(model, device=device)
        self.optimizer = optimizer
        self.thresh = thresh
        self.thresh_end = thresh - thresh_gap
        self.thresh_des = thresh_des
        self.temp = temp
        self.buffer_size = buffer_size
        self.samples_buffer: Optional[torch.Tensor] = None
        self._reset_state_captured = False

    def capture_reset_state(self) -> None:
        self._initial_model_state = deepcopy(self.model.state_dict())
        self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())
        self._initial_thresh = self.thresh
        if hasattr(self, "supports"):
            self._initial_supports = self.supports.detach().clone()
            self._initial_labels = self.labels.detach().clone()
            self._initial_ent = self.ent.detach().clone()
        self._reset_state_captured = True

    def reset_for_new_sample(self) -> None:
        if not self._reset_state_captured:
            self.capture_reset_state()
        self.model.load_state_dict(self._initial_model_state, strict=True)
        self.optimizer.load_state_dict(self._initial_optimizer_state)
        self.thresh = self._initial_thresh
        self.samples_buffer = None
        if hasattr(self, "_initial_supports"):
            self.supports = self._initial_supports.detach().clone()
            self.labels = self._initial_labels.detach().clone()
            self.ent = self._initial_ent.detach().clone()

    @torch.enable_grad()
    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_sample_num = inputs.shape[0]
        samples = inputs
        if self.samples_buffer is not None:
            samples = torch.cat((samples, self.samples_buffer), dim=0)

        logits = predict_logits(self.model, samples)
        probs = F.softmax(logits, dim=1)
        probs_des, _ = torch.sort(probs, descending=True)

        margins = probs_des[:, 0] - probs_des[:, 1]
        mask_hard = margins > self.thresh
        mask_unselect = (probs_des[:, 0] - probs_des[:, -1]) < self.thresh
        mask_partial = ~(mask_hard | mask_unselect)

        if mask_unselect.any():
            _, idxs = torch.sort(margins[mask_unselect], descending=True)
            unselected = samples[mask_unselect][idxs]
            self.samples_buffer = unselected[: self.buffer_size]
        else:
            self.samples_buffer = None

        self._optimize_batch(logits, probs, probs_des, mask_hard, mask_partial)

        if self.thresh > self.thresh_end:
            self.thresh -= self.thresh_des

        return logits[:origin_sample_num].detach()

    def _optimize_batch(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        probs_des: torch.Tensor,
        mask_hard: torch.Tensor,
        mask_partial: torch.Tensor,
    ) -> None:
        hard_count = int(mask_hard.sum().item())
        partial_count = int(mask_partial.sum().item())
        if hard_count + partial_count == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss_hard = logits.new_tensor(0.0)
        if hard_count > 0:
            loss_hard = F.cross_entropy(
                logits[mask_hard] / self.temp,
                logits[mask_hard].detach().argmax(dim=1),
            )

        loss_partial = logits.new_tensor(0.0)
        if partial_count > 0:
            partial_labels = (
                (probs[mask_partial] + self.thresh)
                > probs_des[mask_partial][:, 0].reshape(-1, 1)
            ).float()
            loss_partial = cc_loss(logits[mask_partial], partial_labels.detach(), self.temp)

        lam_hard = hard_count / (hard_count + partial_count)
        loss = loss_hard * lam_hard + loss_partial * (1.0 - lam_hard)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


class PASLE_E(PASLE):
    """PASLE with prototype-based label filtering for domain generalization."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        num_classes: int,
        thresh: float = 0.6,
        thresh_gap: float = 0.1,
        thresh_des: float = 0.001,
        temp: float = 3.0,
        buffer_size: int = 32,
        filter_k: int = 100,
    ):
        super().__init__(
            model=model,
            device=device,
            optimizer=optimizer,
            thresh=thresh,
            thresh_gap=thresh_gap,
            thresh_des=thresh_des,
            temp=temp,
            buffer_size=buffer_size,
        )
        self.num_classes = num_classes
        self.filter_k = filter_k
        self.model_copy = deepcopy(model).eval()
        for p in self.model_copy.parameters():
            p.requires_grad = False

        warmup_supports = get_classifier_weight(self.model_copy)
        self.warmup_supports = warmup_supports
        warmup_prob = classify_features(self.model_copy, warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(
            warmup_prob.argmax(dim=1), num_classes=self.num_classes
        ).float()
        self.supports = self.warmup_supports.data.clone()
        self.labels = self.warmup_labels.data.clone()
        self.ent = self.warmup_ent.data.clone()

    @torch.enable_grad()
    def adapt_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        self.origin_sample_num = inputs.shape[0]
        samples = inputs
        if self.samples_buffer is not None:
            samples = torch.cat((samples, self.samples_buffer), dim=0)

        logits = predict_logits(self.model, samples)
        probs = F.softmax(logits, dim=1)
        probs_des, _ = torch.sort(probs, descending=True)

        margins = probs_des[:, 0] - probs_des[:, 1]
        mask_hard = margins > self.thresh
        mask_unselect = (probs_des[:, 0] - probs_des[:, -1]) < self.thresh
        mask_partial = ~(mask_hard | mask_unselect)

        if mask_unselect.any():
            _, idxs = torch.sort(margins[mask_unselect], descending=True)
            unselected = samples[mask_unselect][idxs]
            self.samples_buffer = unselected[: self.buffer_size]
        else:
            self.samples_buffer = None

        partial_labels = None
        if mask_partial.any():
            partial_labels = (
                (probs[mask_partial] + self.thresh)
                > probs_des[mask_partial][:, 0].reshape(-1, 1)
            ).float()

        label_prototype = self._get_label_with_prototype(samples).argmax(dim=1)

        mask_hard_same = torch.zeros_like(mask_hard)
        if mask_hard.any():
            mask_hard_same[mask_hard] = logits[mask_hard].argmax(dim=1) == label_prototype[mask_hard]

        mask_partial_same = torch.zeros(mask_partial.sum(), dtype=torch.bool, device=logits.device)
        if mask_partial.any() and partial_labels is not None:
            rows = torch.arange(mask_partial.long().sum(), device=logits.device)
            cols = label_prototype[mask_partial]
            mask_partial_same = partial_labels[rows, cols] > 0

        self._optimize_filtered_batch(
            logits=logits,
            mask_hard=mask_hard,
            mask_partial=mask_partial,
            mask_hard_same=mask_hard_same,
            mask_partial_same=mask_partial_same,
            partial_labels=partial_labels,
        )

        if self.thresh > self.thresh_end:
            self.thresh -= self.thresh_des

        return logits[: self.origin_sample_num].detach()

    def _optimize_filtered_batch(
        self,
        logits: torch.Tensor,
        mask_hard: torch.Tensor,
        mask_partial: torch.Tensor,
        mask_hard_same: torch.Tensor,
        mask_partial_same: torch.Tensor,
        partial_labels: Optional[torch.Tensor],
    ) -> None:
        hard_count = int(mask_hard_same.sum().item())
        partial_count = int(mask_partial_same.sum().item())
        if hard_count + partial_count == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss_hard = logits.new_tensor(0.0)
        if hard_count > 0:
            hard_logits = logits[mask_hard][mask_hard_same[mask_hard]]
            loss_hard = F.cross_entropy(
                hard_logits / self.temp,
                hard_logits.detach().argmax(dim=1),
            )

        loss_partial = logits.new_tensor(0.0)
        if partial_count > 0 and partial_labels is not None:
            partial_logits = logits[mask_partial][mask_partial_same]
            partial_targets = partial_labels[mask_partial_same]
            loss_partial = cc_loss(partial_logits, partial_targets.detach(), self.temp)

        lam_hard = hard_count / (hard_count + partial_count)
        loss = loss_hard * lam_hard + loss_partial * (1.0 - lam_hard)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def _get_label_with_prototype(self, x: torch.Tensor) -> torch.Tensor:
        z = extract_features(self.model_copy, x)
        p = classify_features(self.model_copy, z)
        yhat = F.one_hot(p.argmax(dim=1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)

        current_n = self.origin_sample_num
        self.supports = torch.cat([self.supports, z[:current_n]])
        self.labels = torch.cat([self.labels, yhat[:current_n]])
        self.ent = torch.cat([self.ent, ent[:current_n]])

        supports, labels = self._select_supports()
        supports = F.normalize(supports, dim=1)
        weights = supports.T @ labels
        return z @ F.normalize(weights, dim=0)

    def _select_supports(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        if self.filter_k == -1:
            return self.supports, self.labels

        indices = []
        all_indices = torch.arange(len(ent_s), device=ent_s.device)
        for class_id in range(self.num_classes):
            class_mask = y_hat == class_id
            if not class_mask.any():
                continue
            class_indices = all_indices[class_mask]
            _, order = torch.sort(ent_s[class_mask])
            selected = class_indices[order][: self.filter_k]
            indices.append(selected)
        if not indices:
            return self.supports, self.labels

        indices = torch.cat(indices)
        return self.supports[indices], self.labels[indices]


def build_pasle_method(
    model: nn.Module,
    device: torch.device,
    num_classes: int,
    variant: str,
    update_param: str,
    lr: float,
    thresh: float,
    thresh_gap: float,
    thresh_des: float,
    temp: float,
    buffer_size: int,
    filter_k: int,
) -> PASLE:
    model = configure_model_for_pasle(model, update_param=update_param)
    if update_param == "all":
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params, _ = collect_affine_params(model)

    if not params:
        raise RuntimeError("PASLE found no trainable parameters.")

    optimizer = optim.Adam(params, lr=lr)
    common_kwargs = dict(
        model=model,
        device=device,
        optimizer=optimizer,
        thresh=thresh,
        thresh_gap=thresh_gap,
        thresh_des=thresh_des,
        temp=temp,
        buffer_size=buffer_size,
    )
    if variant == "pasle_e":
        method = PASLE_E(num_classes=num_classes, filter_k=filter_k, **common_kwargs)
    else:
        method = PASLE(**common_kwargs)
    method.capture_reset_state()
    return method


def run_pasle_evaluation(
    loaders,
    model_cfg: ModelConfig,
    model_name: str,
    device: torch.device,
    num_classes: int,
    variant: str,
    update_param: str = "all",
    lr: float = 1e-4,
    thresh: float = 0.6,
    thresh_gap: float = 0.1,
    thresh_des: float = 0.001,
    temp: float = 3.0,
    buffer_size: int = 16,
    filter_k: int = 100,
    reset_each_sample: bool = False,
) -> float:
    """Run one PASLE evaluation and return test accuracy."""
    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    pasle = build_pasle_method(
        model=model,
        device=device,
        num_classes=num_classes,
        variant=variant,
        update_param=update_param,
        lr=lr,
        thresh=thresh,
        thresh_gap=thresh_gap,
        thresh_des=thresh_des,
        temp=temp,
        buffer_size=buffer_size,
        filter_k=filter_k,
    )
    metrics = pasle.predict_loader(loaders["test"], reset_each_sample=reset_each_sample)
    return float(metrics.get("acc", 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQ 数据集 PASLE (Selective Label Enhancement Learning) TTA",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["wdcnn", "resnet18", "resnet34", "resnet50", "resnet101"],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument(
        "--test_speeds",
        type=str,
        default="2,3",
        help="目标域测试工况 speeds",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--update_param",
        type=str,
        default="all",
        choices=["all", "affine"],
        help="可训练参数范围，对齐官方 PASLE",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "pasle", "pasle_e"],
        help="PASLE 变体：corruption 用 pasle，域偏移用 pasle_e",
    )
    parser.add_argument("--thresh", type=float, default=-1.0, help="初始置信度阈值，<0 则自动选择")
    parser.add_argument("--thresh_gap", type=float, default=0.1)
    parser.add_argument("--thresh_des", type=float, default=0.001)
    parser.add_argument("--temp", type=float, default=-1.0, help="温度，<0 则自动选择")
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=-1,
        help="未选中样本 buffer，<0 则取 batch_size//4",
    )
    parser.add_argument("--filter_k", type=int, default=100, help="PASLE_E 每类保留 support 数")
    parser.add_argument("--no_transform", action="store_true")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--corruption_type", type=str, default=None, choices=["noise", "missing"])
    parser.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    return parser.parse_args()


def resolve_variant_and_hparams(args: argparse.Namespace) -> Tuple[str, float, float]:
    if args.variant != "auto":
        variant = args.variant
    elif args.corruption_type:
        variant = "pasle"
    else:
        variant = "pasle_e"

    if args.thresh >= 0:
        thresh = args.thresh
    else:
        thresh = 0.8 if variant == "pasle" else 0.6

    if args.temp >= 0:
        temp = args.temp
    else:
        temp = 5.0 if variant == "pasle" else 3.0

    return variant, thresh, temp


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    variant, thresh, temp = resolve_variant_and_hparams(args)
    buffer_size = args.buffer_size if args.buffer_size > 0 else max(args.batch_size // 4, 1)

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

    pasle_model = build_model(model_cfg, model_name=args.model, device=device, track_running_stats=True)
    pasle = build_pasle_method(
        model=pasle_model,
        device=device,
        num_classes=num_classes,
        variant=variant,
        update_param=args.update_param,
        lr=args.lr,
        thresh=thresh,
        thresh_gap=args.thresh_gap,
        thresh_des=args.thresh_des,
        temp=temp,
        buffer_size=buffer_size,
        filter_k=args.filter_k,
    )
    trainable = sum(p.numel() for p in pasle.model.parameters() if p.requires_grad)
    print(
        f"\n=== PASLE ({variant}, lr={args.lr}, update_param={args.update_param}, "
        f"thresh={thresh}, temp={temp}, buffer_size={buffer_size}) ==="
    )
    print(f"[INFO] PASLE trainable scalars: {trainable}")
    metrics = pasle.predict_loader(loaders["test"])
    print(f"PASLE Acc: {metrics.get('acc', 0.0):.4%}")


if __name__ == "__main__":
    main()
