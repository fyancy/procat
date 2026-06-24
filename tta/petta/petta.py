"""PeTTA: Persistent Test-time Adaptation (NeurIPS 2024).

Ported from https://github.com/hthieu166/petta official implementation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from ..common import TTABase
    from ..test_rotta import ROTTA_RUNTIME_PROTOCOLS, configure_rotta_model, make_optimizer, make_rotta_strong_view
except ImportError:
    from common import TTABase
    from test_rotta import ROTTA_RUNTIME_PROTOCOLS, configure_rotta_model, make_optimizer, make_rotta_strong_view

from .losses import self_training, softmax_entropy
from .memory import DivergenceScore, PeTTAMemory, PrototypeMemory


DEFAULT_PETTA_LR = 1e-3
DEFAULT_PETTA_STEPS = 1
DEFAULT_PETTA_MEMORY_SIZE = 64
DEFAULT_PETTA_UPDATE_FREQUENCY = 64
DEFAULT_PETTA_ALPHA_0 = 0.001
DEFAULT_PETTA_LAMBDA_0 = 10.0
DEFAULT_PETTA_AL_WGT = 1.0
DEFAULT_PETTA_REGULARIZER = "cosine"
DEFAULT_PETTA_LOSS_FUNC = "sce"
DEFAULT_PETTA_RBN_ALPHA = 0.05
DEFAULT_PETTA_NU = 0.001
DEFAULT_PETTA_LAMBDA_T = 1.0
DEFAULT_PETTA_LAMBDA_U = 1.0
DEFAULT_PETTA_STRONG_NOISE_STD = 0.005
DEFAULT_PETTA_SOURCE_PERCENTAGE = 1.0
DEFAULT_PETTA_PROTO_NU = 0.05


class FeatureHead(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(x)


class ClassifierHead(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classify_features(features)


class PeTTA(TTABase):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        source_loader=None,
        lr: float = DEFAULT_PETTA_LR,
        optimizer_name: str = "adam",
        weight_decay: float = 0.0,
        steps: int = DEFAULT_PETTA_STEPS,
        protocol: Optional[str] = None,
        memory_size: int = DEFAULT_PETTA_MEMORY_SIZE,
        update_frequency: int = DEFAULT_PETTA_UPDATE_FREQUENCY,
        alpha_0: float = DEFAULT_PETTA_ALPHA_0,
        lambda_0: float = DEFAULT_PETTA_LAMBDA_0,
        al_wgt: float = DEFAULT_PETTA_AL_WGT,
        regularizer: str = DEFAULT_PETTA_REGULARIZER,
        loss_func: str = DEFAULT_PETTA_LOSS_FUNC,
        adaptive_lambda: bool = True,
        adaptive_alpha: bool = True,
        rbn_alpha: float = DEFAULT_PETTA_RBN_ALPHA,
        lambda_t: float = DEFAULT_PETTA_LAMBDA_T,
        lambda_u: float = DEFAULT_PETTA_LAMBDA_U,
        strong_noise_std: float = DEFAULT_PETTA_STRONG_NOISE_STD,
        source_percentage: float = DEFAULT_PETTA_SOURCE_PERCENTAGE,
        proto_cache_dir: Optional[str] = None,
        proto_cache_tag: str = "resnet18_sq",
        reset_each_sample: bool = False,
        adapt_mode: str = "batch",
        online_batch_size: int = 64,
    ) -> None:
        if protocol:
            protocol = protocol.lower()
            if protocol not in ROTTA_RUNTIME_PROTOCOLS:
                raise ValueError(f"protocol must be one of: {sorted(ROTTA_RUNTIME_PROTOCOLS)}")
            adapt_mode = "batch" if protocol in {"online-batch", "online-batch-bias"} else "sample"
            reset_each_sample = protocol == "standalone"
        if regularizer not in {"l2", "cosine", "none"}:
            raise ValueError("regularizer must be one of: l2, cosine, none")
        if loss_func not in {"sce", "ce"}:
            raise ValueError("loss_func must be one of: sce, ce")

        super().__init__(model, device=device)
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self.optimizer_name = optimizer_name
        self.weight_decay = float(weight_decay)
        self.steps = int(steps)
        self.protocol = protocol
        self.memory_size = int(memory_size)
        self.update_frequency = int(update_frequency)
        self.alpha_0 = float(alpha_0)
        self.lambda_0 = float(lambda_0)
        self.al_wgt = float(al_wgt)
        self.regularizer = regularizer
        self.loss_func = loss_func
        self.adaptive_lambda = bool(adaptive_lambda)
        self.adaptive_alpha = bool(adaptive_alpha)
        self.strong_noise_std = float(strong_noise_std)
        self.source_percentage = float(source_percentage)
        self.reset_each_sample = bool(reset_each_sample)
        self.adapt_mode = adapt_mode
        self.online_batch_size = max(int(online_batch_size), 1)
        self.alpha = self.alpha_0
        self.step = 0

        self.params = configure_rotta_model(self.model, alpha=rbn_alpha, adapt_params="affine", use_robust_bn=True)
        self.optimizer = make_optimizer(self.params, optimizer_name, lr, weight_decay)

        self.model_feat = FeatureHead(self.model).to(self.device)
        self.model_clsf = ClassifierHead(self.model).to(self.device)

        self.model_ema = copy.deepcopy(self.model).to(self.device)
        for param in self.model_ema.parameters():
            param.detach_()
        self.model_ema_feat = FeatureHead(self.model_ema).to(self.device)
        self.model_ema_clsf = ClassifierHead(self.model_ema).to(self.device)

        self.model_init = copy.deepcopy(self.model).to(self.device)
        for param in self.model_init.parameters():
            param.detach_()
        self.model_init_feat = FeatureHead(self.model_init).to(self.device)
        self.model_init_clsf = ClassifierHead(self.model_init).to(self.device)
        self.init_model_state = copy.deepcopy(self.model_init.state_dict())

        cache_dir = Path(proto_cache_dir or "tta/results/petta_prototypes")
        src_feat_mean, src_feat_cov = self._compute_source_features(source_loader, cache_dir, proto_cache_tag)
        src_feat_mean = src_feat_mean.to(self.device)
        src_feat_cov = src_feat_cov.to(self.device)
        self.sample_mem = PeTTAMemory(
            capacity=self.memory_size,
            num_class=self.num_classes,
            lambda_t=lambda_t,
            lambda_u=lambda_u,
        )
        self.proto_mem = PrototypeMemory(src_feat_mean, self.num_classes)
        self.proto_mem.mem_proto = self.proto_mem.mem_proto.to(self.device)
        self.proto_mem.src_proto = self.proto_mem.src_proto.to(self.device)
        self.divg_score = DivergenceScore(src_feat_mean, src_feat_cov).to(self.device)

        self.initial_model_state = copy.deepcopy(self.model.state_dict())
        self.initial_ema_state = copy.deepcopy(self.model_ema.state_dict())
        self.initial_init_state = copy.deepcopy(self.model_init.state_dict())
        self.initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.initial_memory = self.sample_mem.clone()
        self.initial_proto_mem = self.proto_mem.clone()

    def reset(self) -> None:
        self.model.load_state_dict(self.initial_model_state, strict=True)
        self.model_ema.load_state_dict(self.initial_ema_state, strict=True)
        self.model_init.load_state_dict(self.initial_init_state, strict=True)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        self.sample_mem = self.initial_memory.clone()
        self.proto_mem = self.initial_proto_mem.clone()
        self.proto_mem.mem_proto = self.proto_mem.mem_proto.to(self.device)
        self.proto_mem.src_proto = self.proto_mem.src_proto.to(self.device)
        self.alpha = self.alpha_0
        self.step = 0

    def reset_for_new_sample(self) -> None:
        self.reset()

    @staticmethod
    def update_ema_variables(ema_model: nn.Module, model: nn.Module, alpha: float) -> nn.Module:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1.0 - alpha) * ema_param.data[:] + alpha * param.data[:]
        return ema_model

    def _proto_cache_paths(self, cache_dir: Path, tag: str) -> Tuple[Path, Path]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = f"protos_sq_{tag}_pct{self.source_percentage:g}"
        return cache_dir / f"{stem}_mean.pt", cache_dir / f"{stem}_cov.pt"

    @torch.no_grad()
    def _compute_source_features(
        self,
        source_loader,
        cache_dir: Path,
        tag: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_path, cov_path = self._proto_cache_paths(cache_dir, tag)
        if mean_path.is_file() and cov_path.is_file():
            return torch.load(mean_path, map_location="cpu"), torch.load(cov_path, map_location="cpu")

        if source_loader is None:
            raise ValueError("PeTTA requires a source_loader to compute class prototypes.")

        features_src = []
        labels_src = []
        self.model.eval()
        max_samples = 100_000
        collected = 0
        for inputs, targets in source_loader:
            inputs = inputs.to(self.device, non_blocking=True).float()
            tmp_features = self.model_feat(inputs)
            preds = self.model_clsf(tmp_features).argmax(1).cpu()
            features_src.append(tmp_features.detach().cpu())
            labels_src.append(preds)
            collected += int(inputs.size(0))
            if collected >= max_samples:
                break

        features_src = torch.cat(features_src, dim=0)
        labels_src = torch.cat(labels_src, dim=0)
        if self.source_percentage < 1.0:
            keep = max(1, int(features_src.size(0) * self.source_percentage))
            features_src = features_src[:keep]
            labels_src = labels_src[:keep]

        src_feat_mean = []
        src_feat_cov = []
        for cls in range(self.num_classes):
            mask = labels_src == cls
            if mask.any():
                cls_feats = features_src[mask]
                src_feat_mean.append(cls_feats.mean(dim=0, keepdim=True))
                src_feat_cov.append(torch.diagonal(cls_feats.T.cov()).unsqueeze(0))
            else:
                src_feat_mean.append(torch.zeros(1, features_src.size(1)))
                src_feat_cov.append(torch.ones(1, features_src.size(1)))

        src_feat_mean_t = torch.cat(src_feat_mean, dim=0)
        src_feat_cov_t = torch.cat(src_feat_cov, dim=0)
        torch.save(src_feat_mean_t, mean_path)
        torch.save(src_feat_cov_t, cov_path)
        return src_feat_mean_t, src_feat_cov_t

    def regularization_loss(self, model: nn.Module) -> torch.Tensor:
        if self.regularizer == "none":
            return torch.zeros((), device=self.device)
        reg_lss = torch.zeros((), device=self.device)
        count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            init_param = self.init_model_state[name].to(self.device)
            if self.regularizer == "l2":
                reg_lss = reg_lss + ((param - init_param) ** 2).sum()
            else:
                reg_lss = reg_lss - F.cosine_similarity(param[None, ...], init_param[None, ...]).mean()
            count += 1
        return reg_lss / max(count, 1)

    @torch.enable_grad()
    def _adapt_from_memory(self, pseudo_labels: torch.Tensor, ema_sup_feat: torch.Tensor) -> None:
        sup_data, _ = self.sample_mem.get_memory()
        if not sup_data:
            return

        sup_data_tensor = torch.stack(sup_data).to(self.device)
        self.model_ema.train()
        ema_feat = self.model_ema_feat(sup_data_tensor)
        x_ema = self.model_ema_clsf(ema_feat)

        self.model.train()
        self.model_init.train()
        p_ori = self.model(sup_data_tensor)
        init_feat = self.model_init_feat(sup_data_tensor)
        init_model_out = self.model_init_clsf(init_feat)
        strong_sup_aug = make_rotta_strong_view(sup_data_tensor, noise_std=self.strong_noise_std)
        stu_sup_feat = self.model_feat(strong_sup_aug)
        p_aug = self.model_clsf(stu_sup_feat)

        if self.loss_func == "sce":
            cls_lss = self_training(p_ori, p_aug, x_ema).mean()
        else:
            cls_lss = softmax_entropy(p_aug, x_ema).mean()

        reg_lss = self.regularization_loss(self.model)
        anchor_lss = softmax_entropy(p_aug, init_model_out).mean()
        reg_wgt = self.lambda_0

        if self.adaptive_lambda or self.adaptive_alpha:
            lbl_uniq = torch.unique(pseudo_labels)
            divg_scr = 1.0 - torch.exp(-self.divg_score(feats=self.proto_mem.mem_proto[lbl_uniq], pseudo_lbls=lbl_uniq))
            self.proto_mem.update(feats=ema_sup_feat.detach(), pseudo_lbls=pseudo_labels)
            if self.adaptive_lambda:
                reg_wgt = float(divg_scr.mean().item()) * self.lambda_0
            if self.adaptive_alpha:
                self.alpha = float((1.0 - divg_scr.mean()).item()) * self.alpha_0

        total_lss = cls_lss + reg_wgt * reg_lss + self.al_wgt * anchor_lss
        self.optimizer.zero_grad(set_to_none=True)
        total_lss.backward()
        self.optimizer.step()
        self.update_ema_variables(self.model_ema, self.model, self.alpha)

    @torch.no_grad()
    def _teacher_logits(self, batch_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model_ema.eval()
        ema_sup_feat = self.model_ema_feat(batch_data)
        p_ema = self.model_ema_clsf(ema_sup_feat)
        predict = torch.softmax(p_ema, dim=1)
        pseudo_lbls = torch.argmax(predict, dim=1)
        entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)
        return p_ema, ema_sup_feat, pseudo_lbls, entropy

    def forward_and_adapt(self, batch_data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.step += 1
        p_ema, ema_sup_feat, pseudo_lbls, entropy = self._teacher_logits(batch_data)

        for idx, sample in enumerate(batch_data):
            true_label = int(labels[idx].item()) if labels is not None else -1
            self.sample_mem.add_instance(
                (sample.detach(), int(pseudo_lbls[idx].item()), float(entropy[idx].item()), true_label)
            )

        if self.sample_mem.get_occupancy() > 0:
            for _ in range(self.steps):
                self._adapt_from_memory(pseudo_lbls, ema_sup_feat)
        return p_ema.detach()

    def predict_loader(
        self,
        data_loader,
        targets=None,
        reset_each_sample: bool = False,
    ):
        import numpy as np

        self.model.eval()
        total_correct = 0
        total_samples = 0
        for inputs, labels in data_loader:
            inputs = torch.from_numpy(inputs) if isinstance(inputs, np.ndarray) else inputs
            labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
            inputs = inputs.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True).long()
            if reset_each_sample:
                batch_logits = []
                for sample_idx in range(inputs.size(0)):
                    self.reset_for_new_sample()
                    batch_logits.append(self.adapt_one_batch(inputs[sample_idx : sample_idx + 1], labels[sample_idx : sample_idx + 1]))
                logits = torch.cat(batch_logits, dim=0)
            else:
                logits = self.adapt_one_batch(inputs, labels)

            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.numel())

        metrics = {}
        if total_samples > 0:
            metrics["acc"] = total_correct / total_samples
        return metrics

    def adapt_one_batch(self, inputs: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.reset_each_sample:
            self.reset()

        if self.adapt_mode == "batch":
            logits_parts: List[torch.Tensor] = []
            for start in range(0, inputs.size(0), self.online_batch_size):
                chunk = inputs[start : start + self.online_batch_size]
                chunk_labels = labels[start : start + self.online_batch_size] if labels is not None else None
                logits_parts.append(self.forward_and_adapt(chunk, chunk_labels))
            return torch.cat(logits_parts, dim=0)

        logits_parts: List[torch.Tensor] = []
        for sample_idx in range(inputs.size(0)):
            if self.reset_each_sample:
                self.reset()
            sample = inputs[sample_idx : sample_idx + 1]
            sample_label = labels[sample_idx : sample_idx + 1] if labels is not None else None
            logits_parts.append(self.forward_and_adapt(sample, sample_label))
        return torch.cat(logits_parts, dim=0)


def run_petta_evaluation(
    loaders,
    model_cfg,
    model_name: str,
    device: torch.device,
    num_classes: int,
    protocol: str = "online-batch",
    lr: float = DEFAULT_PETTA_LR,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    steps: int = DEFAULT_PETTA_STEPS,
    memory_size: int = DEFAULT_PETTA_MEMORY_SIZE,
    update_frequency: int = DEFAULT_PETTA_UPDATE_FREQUENCY,
    alpha_0: float = DEFAULT_PETTA_ALPHA_0,
    lambda_0: float = DEFAULT_PETTA_LAMBDA_0,
    al_wgt: float = DEFAULT_PETTA_AL_WGT,
    regularizer: str = DEFAULT_PETTA_REGULARIZER,
    loss_func: str = DEFAULT_PETTA_LOSS_FUNC,
    adaptive_lambda: bool = True,
    adaptive_alpha: bool = True,
    rbn_alpha: float = DEFAULT_PETTA_RBN_ALPHA,
    lambda_t: float = DEFAULT_PETTA_LAMBDA_T,
    lambda_u: float = DEFAULT_PETTA_LAMBDA_U,
    strong_noise_std: float = DEFAULT_PETTA_STRONG_NOISE_STD,
    source_percentage: float = DEFAULT_PETTA_SOURCE_PERCENTAGE,
    proto_cache_tag: str = "resnet18_sq",
    online_batch_size: int = 64,
) -> float:
    try:
        from ..common import build_model
    except ImportError:
        from common import build_model

    model = build_model(model_cfg, model_name=model_name, device=device, track_running_stats=True)
    adapter = PeTTA(
        model=model,
        device=device,
        num_classes=num_classes,
        source_loader=loaders.get("train"),
        protocol=protocol,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        steps=steps,
        memory_size=memory_size,
        update_frequency=update_frequency,
        alpha_0=alpha_0,
        lambda_0=lambda_0,
        al_wgt=al_wgt,
        regularizer=regularizer,
        loss_func=loss_func,
        adaptive_lambda=adaptive_lambda,
        adaptive_alpha=adaptive_alpha,
        rbn_alpha=rbn_alpha,
        lambda_t=lambda_t,
        lambda_u=lambda_u,
        strong_noise_std=strong_noise_std,
        source_percentage=source_percentage,
        proto_cache_tag=proto_cache_tag,
        online_batch_size=online_batch_size,
    )
    return float(adapter.predict_loader(loaders["test"]).get("acc", 0.0))
