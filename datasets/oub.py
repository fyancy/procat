"""OUB (Ottawa University Bearing, time-varying speed) dataset loader."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.paths_oub import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_DIR,
    DOMAIN_CODES,
    FAULT_FOLDERS,
    get_oub_mat_path,
)
from utils.data_utils import gen_samples_from_data
from utils.ts_transform import transform_value

NUM_DOMAINS = 4
NUM_CLASSES = 5


def load_vibration_channel(mat_path: Path) -> np.ndarray:
    data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "Channel_1" not in data:
        raise KeyError(f"Channel_1 not found in {mat_path}")
    signal = np.asarray(data["Channel_1"], dtype=np.float32).reshape(-1)
    return signal


class OUBGenerator:
    def __init__(
        self,
        raw_dir: str | Path | None = None,
        x_length: int = 2048,
        sample_overlap: float = 0.65,
        samples_per_class: int = 100,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
        self.sample_len = int(x_length)
        self.sample_overlap = float(sample_overlap)
        self.samples_per_class = int(samples_per_class)

    def _slice_file(self, mat_path: Path, n_samples: int) -> np.ndarray:
        signal = load_vibration_channel(mat_path)
        samples = gen_samples_from_data(
            signal,
            sample_length=self.sample_len,
            overlap=self.sample_overlap,
            num_samples=n_samples,
        )
        return samples.astype(np.float32)

    def _samples_for_domain_class(self, domain: int, label: int) -> np.ndarray:
        per_trial = max(1, int(np.ceil(self.samples_per_class / 3)))
        chunks = []
        for trial in (1, 2, 3):
            mat_path = get_oub_mat_path(self.raw_dir, label, domain, trial)
            if not mat_path.exists():
                raise FileNotFoundError(f"Missing OUB file: {mat_path}")
            chunks.append(self._slice_file(mat_path, per_trial))
        merged = np.concatenate(chunks, axis=0)
        if merged.shape[0] < self.samples_per_class:
            raise ValueError(
                f"Not enough samples for domain={domain}, label={label}: "
                f"got {merged.shape[0]}, need {self.samples_per_class}"
            )
        return merged[: self.samples_per_class]

    def data_init(self) -> Tuple[np.ndarray, np.ndarray]:
        cache_path = DEFAULT_CACHE_DIR / (
            f"oub_len{self.sample_len}_n{self.samples_per_class}_overlap{self.sample_overlap:.2f}.npy"
        )
        if cache_path.exists():
            xy = np.load(cache_path, allow_pickle=True).item()
            return xy["x"].astype(np.float32), xy["y"].astype(np.int32)

        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"OUB raw dir not found: {self.raw_dir}. "
                "Run scripts/download_oub.py first."
            )

        x = np.zeros(
            (NUM_DOMAINS, NUM_CLASSES, self.samples_per_class, 1, self.sample_len),
            dtype=np.float32,
        )
        y = np.zeros((NUM_DOMAINS, NUM_CLASSES, self.samples_per_class), dtype=np.int32)
        for domain in range(NUM_DOMAINS):
            for label in range(NUM_CLASSES):
                samples = self._samples_for_domain_class(domain, label)
                x[domain, label, :, 0, :] = samples
                y[domain, label, :] = label

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, {"x": x, "y": y})
        print(f"Saved OUB cache: {cache_path}, x={x.shape}, y={y.shape}")
        return x, y

    def data_gen(self, flatten: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.data_init()
        if flatten:
            x = x.reshape(-1, 1, self.sample_len)
            y = y.reshape(-1)
        return x, y


def get_oub_data(
    train_ratio: float = 0.8,
    train_domains: Tuple[int, ...] = (0,),
    test_domains: Tuple[int, ...] = (1, 2, 3),
    x_length: int = 2048,
    samples_per_class: int = 100,
    *,
    source_domain: Optional[int] = None,
    target_domains: Optional[Sequence[int]] = None,
    train_trials: Sequence[int] = (1, 2),
    test_trials: Sequence[int] = (1,),
    test_samples_per_trial: Optional[int] = 100,
    sample_len: int = 2048,
):
    """Build train/test arrays for one OUB cross-domain task.

    Legacy API: ``train_domains`` / ``test_domains`` / ``train_ratio``.
    Trial-wise API (supps): ``source_domain`` + ``target_domains`` + trial args.
    """
    if source_domain is not None and target_domains is not None:
        from tta.oub.oub_2048_data import get_oub_data_trialwise

        return get_oub_data_trialwise(
            source_domain=int(source_domain),
            target_domains=tuple(int(d) for d in target_domains),
            train_trials=tuple(int(t) for t in train_trials),
            test_trials=tuple(int(t) for t in test_trials),
            sample_len=int(sample_len),
            test_samples_per_trial=test_samples_per_trial,
        )

    generator = OUBGenerator(x_length=x_length, samples_per_class=samples_per_class)
    X, Y = generator.data_gen(flatten=False)
    num_train = int(train_ratio * samples_per_class)

    train_domains = np.asarray(train_domains, dtype=int)
    test_domains = np.asarray(test_domains, dtype=int)
    if train_domains.size == 0 or test_domains.size == 0:
        raise ValueError("train_domains and test_domains must not be empty")

    x_train = X[train_domains, :, :num_train]
    y_train = Y[train_domains, :, :num_train]

    test_x_list = []
    test_y_list = []
    mask_seen = np.isin(test_domains, train_domains)
    seen_domains = test_domains[mask_seen]
    unseen_domains = test_domains[~mask_seen]

    if len(seen_domains) > 0:
        _x = X[seen_domains, :, num_train:]
        _y = Y[seen_domains, :, num_train:]
        test_x_list.append(_x.reshape(-1, 1, x_length))
        test_y_list.append(_y.reshape(-1))

    if len(unseen_domains) > 0:
        _x = X[unseen_domains, :, :]
        _y = Y[unseen_domains, :, :]
        test_x_list.append(_x.reshape(-1, 1, x_length))
        test_y_list.append(_y.reshape(-1))

    if test_x_list:
        x_test = np.concatenate(test_x_list, axis=0).astype(np.float32)
        y_test = np.concatenate(test_y_list, axis=0).astype(np.int32)
    else:
        x_test = np.empty((0, 1, x_length), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.int32)

    x_train = x_train.reshape(-1, 1, x_length).astype(np.float32)
    y_train = y_train.reshape(-1).astype(np.int32)
    return x_train, y_train, x_test, y_test


class OUBDataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        transform: bool = False,
        augment: bool = False,
        in_channels: int = 1,
        domain_ids: Optional[np.ndarray] = None,
        trial_ids: Optional[np.ndarray] = None,
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
        self.trial_ids = (
            np.asarray(trial_ids, dtype=np.int64).reshape(-1)
            if trial_ids is not None
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
