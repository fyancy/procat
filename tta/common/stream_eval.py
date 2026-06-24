"""Online-batch stream evaluation helpers for TTA runners."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def as_tensor(value) -> torch.Tensor:
    return torch.from_numpy(value) if isinstance(value, np.ndarray) else value


def batch_row(
    scenario: str,
    batch_idx: int,
    correct: int,
    samples: int,
    total_correct: int,
    total_samples: int,
    extra: str = "",
    method: str = "",
) -> Dict[str, object]:
    return {
        "scenario": scenario,
        "method": method,
        "extra": extra,
        "batch_idx": batch_idx,
        "batch_size": samples,
        "batch_correct": correct,
        "batch_acc": correct / max(samples, 1),
        "cumulative_correct": total_correct,
        "cumulative_samples": total_samples,
        "cumulative_acc": total_correct / max(total_samples, 1),
    }


def evaluate_stream(
    scenario: str,
    data_loader,
    device: torch.device,
    logits_fn,
    extra: str = "",
    max_batches: int = 0,
) -> Tuple[float, List[Dict[str, object]]]:
    total_correct = 0
    total_samples = 0
    rows: List[Dict[str, object]] = []

    for batch_idx, batch in enumerate(data_loader):
        inputs, labels = batch[0], batch[1]
        inputs = as_tensor(inputs).to(device, non_blocking=True).float()
        labels = as_tensor(labels).to(device, non_blocking=True).long()

        logits = logits_fn(inputs)
        preds = logits.argmax(dim=1)
        correct = int((preds == labels).sum().item())
        samples = int(labels.numel())
        total_correct += correct
        total_samples += samples
        rows.append(
            batch_row(
                scenario=scenario,
                batch_idx=batch_idx,
                correct=correct,
                samples=samples,
                total_correct=total_correct,
                total_samples=total_samples,
                extra=extra,
            )
        )
        if max_batches and (batch_idx + 1) >= max_batches:
            break

    return total_correct / max(total_samples, 1), rows
