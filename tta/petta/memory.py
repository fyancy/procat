"""PeTTA memory and prototype utilities (ported from official petta repo)."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn


def compute_feat_mean(feats: torch.Tensor, pseudo_lbls: torch.Tensor):
    lbl_uniq = torch.unique(pseudo_lbls)
    lbl_group = [torch.where(pseudo_lbls == label)[0] for label in lbl_uniq]
    group_avgs = [feats[idcs].mean(axis=0, keepdim=True) for idcs in lbl_group]
    return lbl_uniq, group_avgs


class DivergenceScore(nn.Module):
    def __init__(self, src_prototype: torch.Tensor, src_prototype_cov: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("src_proto", src_prototype)
        self.register_buffer("src_proto_cov", src_prototype_cov)

    def forward(self, feats: torch.Tensor, pseudo_lbls: torch.Tensor) -> torch.Tensor:
        lbl_uniq, group_avgs = compute_feat_mean(feats, pseudo_lbls)
        stacked = torch.cat(group_avgs, dim=0)
        target = self.src_proto[lbl_uniq]
        target_cov = self.src_proto_cov[lbl_uniq]
        return ((stacked - target).pow(2) / (target_cov + 1e-6)).mean()


class PrototypeMemory:
    def __init__(self, src_prototype: torch.Tensor, num_classes: int) -> None:
        self.src_proto = src_prototype.squeeze(1)
        self.mem_proto = deepcopy(self.src_proto)
        self.num_classes = int(num_classes)

    def update(self, feats: torch.Tensor, pseudo_lbls: torch.Tensor, nu: float = 0.05) -> None:
        lbl_uniq = torch.unique(pseudo_lbls)
        for label in lbl_uniq.tolist():
            idcs = torch.where(pseudo_lbls == label)[0]
            batch_avg = feats[idcs].mean(axis=0)
            self.mem_proto[label] = (1.0 - nu) * self.mem_proto[label] + nu * batch_avg

    def clone(self) -> "PrototypeMemory":
        cloned = PrototypeMemory(self.src_proto.unsqueeze(1), self.num_classes)
        cloned.mem_proto = deepcopy(self.mem_proto)
        return cloned


class MemoryItem:
    def __init__(self, data=None, uncertainty: float = 0.0, age: int = 0, true_label: int = -1) -> None:
        self.data = data
        self.uncertainty = float(uncertainty)
        self.age = int(age)
        self.true_label = int(true_label)

    def increase_age(self) -> None:
        self.age += 1


class PeTTAMemory:
    """Class-balanced memory with true-label bookkeeping (official PeTTAMemory)."""

    def __init__(self, capacity: int, num_class: int, lambda_t: float = 1.0, lambda_u: float = 1.0) -> None:
        self.capacity = int(capacity)
        self.num_class = int(num_class)
        self.per_class = self.capacity / self.num_class
        self.lambda_t = float(lambda_t)
        self.lambda_u = float(lambda_u)
        self.data: List[List[MemoryItem]] = [[] for _ in range(self.num_class)]

    def clone(self) -> "PeTTAMemory":
        return deepcopy(self)

    def get_occupancy(self) -> int:
        return sum(len(class_list) for class_list in self.data)

    def per_class_dist(self) -> List[int]:
        return [len(class_list) for class_list in self.data]

    def heuristic_score(self, age: int, uncertainty: float) -> float:
        return self.lambda_t * (1.0 / (1.0 + math.exp(-age / max(self.capacity, 1)))) + self.lambda_u * uncertainty / max(
            math.log(self.num_class), 1e-8
        )

    def add_age(self) -> None:
        for class_list in self.data:
            for item in class_list:
                item.increase_age()

    def get_majority_classes(self) -> List[int]:
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        return [idx for idx, occupied in enumerate(per_class_dist) if occupied == max_occupied]

    def remove_from_classes(self, classes: List[int], score_base: float) -> bool:
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls
        if max_class is None:
            return True
        if max_score is not None and max_score > score_base:
            self.data[max_class].pop(max_index)
            return True
        return False

    def remove_instance(self, cls: int, score: float) -> bool:
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            return self.remove_from_classes(self.get_majority_classes(), score)
        return self.remove_from_classes([cls], score)

    def add_instance(self, instance: Tuple[torch.Tensor, int, float, int]) -> None:
        x, prediction, uncertainty, true_label = instance
        new_item = MemoryItem(data=x.detach().cpu(), uncertainty=uncertainty, age=0, true_label=true_label)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def get_memory(self) -> Tuple[List[torch.Tensor], List[float]]:
        tmp_data: List[torch.Tensor] = []
        tmp_age: List[float] = []
        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age / max(self.capacity, 1))
        return tmp_data, tmp_age
