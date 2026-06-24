"""
SQ数据集数据文件目录
文件获取举例：inner2['29'][2],
得到的是内圈2下29Hz的第2个文件(actually 3rd)，
默认文件夹内排列顺序，每一个转速下共有3个文件[0~2]
train_dir109 1-故障程度 09-转速(Hz)
train 均使用第0个数据文件，test均使用第1个数据文件
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tsaug
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
from utils.data_utils import sample_split  # , sample_label_shuffle, my_normalization, wgn
from datasets.paths_config import sq_npy_path
from datasets.paths_sq import get_SQ_dir
from utils.ts_transform import transform_value
# from utils.sig_env import batch_signal_autocorrelation, envelope_signal


def load_np_data(f_path, num_data):
    return np.load(f_path)[:num_data]


class SQGenerator(Dataset):
    def __init__(self, x_length=2048, transform=False, resample=True):
        super().__init__()
        self.transform = transform
        self.sample_len = x_length
        self.sample_overlap = 0.65
        self.resample = resample
    
    def data_init(self):
        # if self.resample:
            # file_dir = r"F:\Datasets\SQdata\numpy_data_resampled"  # r"F:\Datasets\SQdata\numpy_data"
        # else:
            # file_dir = r"F:\Datasets\SQdata\numpy_data_unresampled"
        # sq_data_path = Path(file_dir) / rf"sq_wgn_{self.snr}.npy"
        sq_data_path = str(sq_npy_path(resample=self.resample))
        
        if os.path.exists(sq_data_path):
            xy = np.load(sq_data_path, allow_pickle=True).item()
            x, y = xy["x"], xy["y"]
        else:
            data_files_10 = get_SQ_dir(nc=7, speed='09')
            data_files_20 = get_SQ_dir(nc=7, speed='19')
            data_files_30 = get_SQ_dir(nc=7, speed='29')
            data_files_40 = get_SQ_dir(nc=7, speed='39')
            x1, y1 = self.get_data(data_files_10, nx_per_file=100, fs=25600, fr=9)  # 确保数据量足够下采样！
            x2, y2 = self.get_data(data_files_20, nx_per_file=100, fs=25600, fr=19)  # (C, N, 1, L)
            x3, y3 = self.get_data(data_files_30, nx_per_file=100, fs=25600, fr=29)
            x4, y4 = self.get_data(data_files_40, nx_per_file=100, fs=25600, fr=39)
            # x, y = np.concatenate([x1, x2, x3, x4], axis=1), \
            # np.concatenate([y1, y2, y3, y4], axis=1) # [Nc,num_each_way, 1, L], [Nc, num_each_way]
            x, y  = np.stack([x1, x2, x3, x4], axis=0), \
                    np.stack([y1, y2, y3, y4], axis=0)  # [4, Nc, num_each_way, 1, L], [4, Nc, num_each_way]
            
            # NOTE: we don't use the following code to shuffle labels and add noise
            # x, y = sample_label_shuffle(x, y, 1)
            # new_x = np.zeros_like(x)
            # for c in range(x.shape[0]):
            #     for n in range(x.shape[1]):
            #         new_x[c, n, 0] = self.add_noise(x[c, n, 0], self.snr)
            # x = new_x

            np.save(sq_data_path, {"x": x, "y": y})
            print(f"Save resampled SQ data, x: {x.shape}, y: {y.shape}")
        x, y = x.astype(np.float32), y.astype(np.int32)

        return x, y
    
    def get_data(self, data_files, nx_per_file, fs, fr):
        n_way = len(data_files)
        all_data = []
        for i in range(n_way):
            x = load_np_data(data_files[i], 200).reshape(-1)
            d = sample_split(x, self.resample, fs, fr, n_samples=nx_per_file,
                             n_period=4, x_length=self.sample_len, overlap=self.sample_overlap)  # (N, len)
            all_data.append(d)
        
        # 确保所有数据的样本数一致 (取最小值)，防止 np.stack 报错
        min_len = min([len(d) for d in all_data])
        if min_len < nx_per_file:
            all_data = [d[:min_len] for d in all_data]
        
        all_data = np.stack(all_data, axis=0, dtype=np.float32)[:, :, None, :]  # (n_way, n, sample_len)
        label = np.arange(n_way, dtype=np.int32).reshape(n_way, 1)
        label = np.repeat(label, all_data.shape[1], axis=1)  # [n_way, examples]

        return all_data, label  # [Nc,num_each_way, 1, L], [Nc, num_each_way]
    
    def data_gen(self, flatten=False):
        x, y = self.data_init()
        if flatten:
            x = x.reshape(x.shape[0], -1)
            y = y.reshape(-1)
        
        return x, y


def apply_corruption(x, corruption_type, severity):
    """
    对数据应用 Corruption。
    x: (N, 1, L) or (N, L)
    severity: 1-5 (int)
    """
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError(f"severity must be in [1, 5] when corruption is enabled, got {severity}")

    x = x.copy()
    
    # tsaug expects (N, L, C) for multi-channel or (N, L) for single channel.
    # Input x is (N, 1, L) (PyTorch style: N, C, L).
    # We need to transpose to (N, L, 1) for tsaug.
    x_tsaug = x.transpose(0, 2, 1)  # (N, L, 1)
    
    if corruption_type == 'noise':
        # 添加高斯白噪声
        scales = [0.1, 0.2, 0.3, 0.4, 0.5]
        scale = scales[severity - 1]
        x_tsaug = tsaug.AddNoise(scale=scale, prob=1.0).augment(x_tsaug)
        
    elif corruption_type == 'missing':
        # 数据缺失/Dropout
        probs = [0.1, 0.2, 0.3, 0.4, 0.5]
        p = probs[severity - 1]
        # fill=0.0 means replace with 0
        x_tsaug = tsaug.Dropout(p=p, fill=0.0, prob=1.0).augment(x_tsaug)
        
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    
    # Transpose back to (N, 1, L)
    x = x_tsaug.transpose(0, 2, 1)
        
    return x


def dynamic_snr_schedule(num_samples, snr_start=10.0, snr_end=0.0, num_levels=11):
    """Return a stream-ordered SNR schedule, e.g. 10, 9, ..., 0 dB."""
    levels = np.linspace(float(snr_start), float(snr_end), int(num_levels), dtype=np.float32)
    if num_samples <= 0:
        return np.empty((0,), dtype=np.float32)
    bins = np.linspace(0, num_samples, len(levels) + 1, dtype=int)
    schedule = np.empty((num_samples,), dtype=np.float32)
    for idx, level in enumerate(levels):
        schedule[bins[idx] : bins[idx + 1]] = level
    return schedule


def apply_dynamic_snr_noise(x, snr_schedule, seed=0):
    """Add per-sample Gaussian noise at the requested SNR in dB."""
    x = x.astype(np.float32, copy=True)
    snr_schedule = np.asarray(snr_schedule, dtype=np.float32).reshape(-1)
    if len(x) != len(snr_schedule):
        raise ValueError(f"x/snr length mismatch: {len(x)} vs {len(snr_schedule)}")

    rng = np.random.default_rng(int(seed))
    reduce_axes = tuple(range(1, x.ndim))
    signal_power = np.mean(np.square(x), axis=reduce_axes, keepdims=True)
    snr_shape = (-1,) + (1,) * (x.ndim - 1)
    snr_linear = np.power(10.0, snr_schedule.reshape(snr_shape) / 10.0)
    noise_power = signal_power / np.maximum(snr_linear, 1e-12)
    noise = rng.standard_normal(size=x.shape).astype(np.float32) * np.sqrt(noise_power).astype(np.float32)
    return (x + noise).astype(np.float32)


def get_sq_data(
    train_ratio=0.8,
    train_speeds=(0, 1),
    test_speeds=(0, 1, 2, 3),
    corruption_type=None,  # 'noise', 'missing', 'noise_dyn'
    severity=0,            # 1-5
):
    """
    获取 SQ 数据集，支持灵活的工况划分和数据 Corruption。

    Args:
        train_ratio: 训练集比例。每个“训练工况”下前 train_ratio 的数据用于训练。
        train_speeds: 训练集包含的工况索引 (0~3)。
        test_speeds: 测试集包含的工况索引 (0~3)。
        corruption_type: 施加在测试集上的 Corruption 类型 ('noise', 'missing')。
        severity: Corruption 强度 (1-5)。

    Returns:
        x_train, y_train, x_test, y_test
    """
    if corruption_type:
        if tuple(train_speeds) != tuple(test_speeds):
            raise ValueError("corruption experiments require train_speeds and test_speeds to be identical")
        if corruption_type not in ['noise', 'missing', 'noise_dyn']:
            raise ValueError("corruption_type must be 'noise', 'missing', or 'noise_dyn'")
        if int(severity) < 0 or int(severity) > 5:
            raise ValueError(f"severity must be in [0, 5], got {severity}")
        assert train_speeds == test_speeds, "train_speeds 和 test_speeds 不一致"
        assert corruption_type in ['noise', 'missing', 'noise_dyn'], "corruption_type must be 'noise', 'missing', or 'noise_dyn'"
    
    sample_len = 2048
    
    # 检查是否已有缓存的 corrupted 数据
    corruption_file = None
    # Legacy cache path disabled; the active v2 cache below is keyed by split parameters.
    use_legacy_cache = False
    if use_legacy_cache and corruption_type and severity > 0:
        dataset_dir = os.path.dirname(r"H:\Datasets\SQdata\numpy_data_resampled\sq_no_noise_resampled_for_att.npy")
        # 确保目录存在，如果不存在则尝试使用默认路径或者 relative path
        if not os.path.exists(dataset_dir):
             dataset_dir = "datasets/cache"
             os.makedirs(dataset_dir, exist_ok=True)
             
        corruption_file = os.path.join(dataset_dir, f"sq_corrupted_{corruption_type}_{severity}.npy")
        
        if os.path.exists(corruption_file):
            print(f"[INFO] Loading corrupted data from {corruption_file}")
            data_dict = np.load(corruption_file, allow_pickle=True).item()
            # 校验参数是否一致 (简单校验)
            # 这里假设如果文件存在，train/test split 逻辑是一致的
            # 为了严谨，应该在文件名里包含 speeds 信息，或者只缓存 x_test
            # 但由于 get_sq_data 的逻辑是动态切分的，最好的方式是：
            # 1. 正常加载原始数据并切分
            # 2. 如果有缓存的 x_test_corrupted，则替换 x_test
            # 3. 否则，对 x_test 进行 corruption 并保存
            pass # 继续下面的流程

    dataset = SQGenerator(x_length=sample_len)
    X, Y = dataset.data_gen(flatten=False)  # (4, 7, 100, 1, 2048) (4, 7, 100)
    num_per_cls = X.shape[2]
    num_train = int(train_ratio * num_per_cls)

    train_speeds = np.array(train_speeds, dtype=int)
    test_speeds = np.array(test_speeds, dtype=int)
    if train_speeds.size == 0 or test_speeds.size == 0:
        raise ValueError("train_speeds and test_speeds must not be empty")

    max_speed_idx = X.shape[0] - 1
    if (
        train_speeds.min() < 0
        or test_speeds.min() < 0
        or train_speeds.max() > max_speed_idx
        or test_speeds.max() > max_speed_idx
    ):
        raise ValueError(
            f"train_speeds/test_speeds 超出范围，当前最多支持索引 0~{max_speed_idx}"
        )

    # --- 构建训练集 ---
    # 仅使用 train_speeds 中的工况，取前 num_train 个样本
    x_train = X[train_speeds, :, :num_train]
    y_train = Y[train_speeds, :, :num_train]

    # --- 构建测试集 ---
    test_x_list = []
    test_y_list = []

    # 找出哪些 test_speeds 也是 train_speeds (Seen)
    mask_seen = np.isin(test_speeds, train_speeds)
    seen_speeds = test_speeds[mask_seen]
    
    # 找出哪些 test_speeds 是新工况 (Unseen)
    unseen_speeds = test_speeds[~mask_seen]

    # 1. Seen Speeds: 取剩余部分
    if len(seen_speeds) > 0:
        _x = X[seen_speeds, :, num_train:]
        _y = Y[seen_speeds, :, num_train:]
        test_x_list.append(_x.reshape(-1, 1, sample_len))
        test_y_list.append(_y.reshape(-1))

    # 2. Unseen Speeds: 取全部数据
    if len(unseen_speeds) > 0:
        _x = X[unseen_speeds, :, :]
        _y = Y[unseen_speeds, :, :]
        test_x_list.append(_x.reshape(-1, 1, sample_len))
        test_y_list.append(_y.reshape(-1))

    if len(test_x_list) > 0:
        x_test = np.concatenate(test_x_list, axis=0)
        y_test = np.concatenate(test_y_list, axis=0)
    else:
        x_test = np.empty((0, *x_train.shape[1:]))
        y_test = np.empty((0, *y_train.shape[1:]))

    # Reshape 为 (N, 1, L)
    x_train = x_train.reshape(-1, 1, sample_len)
    y_train = y_train.reshape(-1)
    # x_test 已经 reshape 过了 (在 append 时)
    
    # --- Apply Corruption ---
    if corruption_type and (severity > 0 or corruption_type == "noise_dyn"):
        # 为了保证不同实验 (speed split) 下如果恰好用到相同数据的一致性，
        # 最好是把 corruption 应用在整个原始数据集 X 上并缓存整个 X_corrupted。
        # 但这会导致文件巨大。
        # 鉴于您的需求是 "train_speeds 和 test_speeds 一样" 的场景下的 corruption，
        # 或者是不同工况下的 corruption。
        # 最简单的策略：基于当前的 x_test 计算 hash 或直接命名缓存文件。
        # 这里我们采用文件命名包含关键信息的方式：
        # sq_corrupted_{type}_{severity}_tr{train_speeds}_te{test_speeds}.npy
        
        # 构造一个唯一的缓存文件名
        tr_str = "-".join(map(str, train_speeds.tolist()))
        te_str = "-".join(map(str, test_speeds.tolist()))
        ratio_str = str(train_ratio).replace(".", "p")
        if corruption_type == "noise_dyn":
            cache_name = (
                f"sq_corrupted_v2_noise_dyn_snr10to0_levels11"
                f"_len{sample_len}_ratio{ratio_str}_tr{tr_str}_te{te_str}.npy"
            )
        else:
            cache_name = (
                f"sq_corrupted_v2_{corruption_type}_{severity}"
                f"_len{sample_len}_ratio{ratio_str}_tr{tr_str}_te{te_str}.npy"
            )
        
        # 尝试确定缓存目录
        cache_dir = Path(__file__).resolve().parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / cache_name
        
        if os.path.exists(cache_path):
            print(f"[INFO] Loading cached corrupted test data from {cache_path}")
            x_test = np.load(cache_path)
        else:
            print(f"[INFO] Applying corruption: {corruption_type} (severity {severity}) ...")
            if corruption_type == "noise_dyn":
                snr_schedule = dynamic_snr_schedule(len(x_test), 10.0, 0.0, 11)
                x_test = apply_dynamic_snr_noise(x_test, snr_schedule, seed=0)
                np.save(cache_path.with_name(cache_path.stem + "_snr.npy"), snr_schedule)
            else:
                x_test = apply_corruption(x_test, corruption_type, severity)
            # 转换为 float32 确保兼容性
            x_test = x_test.astype(np.float32)
            np.save(cache_path, x_test)
            print(f"[INFO] Saved corrupted test data to {cache_path}")

    return x_train, y_train, x_test, y_test



class SQDataset(Dataset):
    def __init__(self, data, labels, transform=False, augment=False, in_channels=1):
        super().__init__()
        self.transform = transform
        self.augment = augment

        self.data = data.reshape(-1, in_channels, data.shape[-1]).astype(np.float32)
        self.labels = labels.reshape(-1).astype(np.int32)
    
    def __getitem__(self, item):
        return self.transform_fn(self.augment_fn(self.data[item])), self.labels[item]

    def __len__(self):
        return len(self.data)
    
    def transform_fn(self, x, transform_type='maxabs'):
        if self.transform:
            # x = envelope_signal(x, method='hilbert')
            x = transform_value(x, transform_type)
            x = np.clip(x, -1, 1)
        
        return x
    
    def augment_fn(self, x):
        # x = tsaug.Convolve(size=3, prob=0.3).augment(x)  # NOTE: 推荐，如果用squeeze
        if self.augment:
            # x = x if np.random.random() < 0.5 else batch_signal_autocorrelation(x)
            # x = x if np.random.random() < 0.5 else envelope_signal(x, method='peak')
            
            # ***************** 仅推荐以下4种 *****************
            x = -x if np.random.random() < 0.5 else x
            x = tsaug.AddNoise(scale=0.15, prob=0.8).augment(x)
            x = tsaug.Reverse(prob=0.5).augment(x)
            x = tsaug.Dropout(p=0.5, fill=0.0, prob=0.2).augment(x)  # fill='ffill', 0.0 is better
            
            # x = -x if np.random.random() < 0.5 else x
            # x = tsaug.AddNoise(scale=0.15, prob=0.8).augment(x)
            # x = tsaug.Reverse(prob=0.5).augment(x)
            # x = tsaug.Dropout(p=0.5, fill=0., prob=0.2).augment(x)  # fill='ffill', 0.0 is better
            # x = tsaug.TimeWarp(n_speed_change=2, max_speed_ratio=3, prob=0.5).augment(x)  # worse
            # x = tsaug.Resize(size=int(x.shape[-1]*1.5), prob=0.5).augment(x)[..., :x.shape[-1]]  # worse
        
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # data = get_sq_data(train_ratio=0.8, cross_domain=True)

    # exit()

    dataset = SQGenerator(x_length=2048, resample=False)
    X, Y = dataset.data_gen(flatten=False)  # (4, 7, 100, 1, 2048) (4, 7, 100)
    print(X.shape, Y.shape)

    _x1 = X[3, 1, 1, 0]
    _x2 = X[0, 1, 1, 0]
    plt.subplot(2, 1, 1)
    plt.plot(_x1)
    plt.subplot(2, 1, 2)
    plt.plot(_x2)
    plt.show()

