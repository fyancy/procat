"""Bogie (转向架齿轮轴承) PyTorch dataset."""

from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from utils.ts_transform import transform_value


class BogieDataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        transform: bool = False,
        augment: bool = False,
        in_channels: int = 1,
        domain_ids: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.transform = transform
        self.augment = augment
        self.data = np.asarray(data, dtype=np.float32).reshape(-1, in_channels, data.shape[-1])
        self.labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        n = len(self.labels)
        self.domain_ids = (
            np.asarray(domain_ids, dtype=np.int64).reshape(-1)
            if domain_ids is not None
            else np.zeros(n, dtype=np.int64)
        )

    def __getitem__(self, item):
        x = self.data[item]
        if self.transform:
            x = transform_value(x, "maxabs")
            x = np.clip(x, -1, 1)
        return x, self.labels[item]

    def __len__(self):
        return len(self.data)
