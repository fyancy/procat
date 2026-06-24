"""转向架齿轮轴承数据集：从 CSV 重建 h5 / npy。

协议（每转速 rpm ∈ {1000, 1500, 2000}）：
- 轴承 12 kHz：每 (类, 载荷) 固定 3 个 CSV (idx 0..2)，每文件 20 窗，不下采样；
  载荷 0/20/40 合并；故障 l/m/h 合并为 6 类。
- 齿轮 24 kHz：每 (类, 载荷) 固定 6 个 CSV (idx 0..5)，每文件 10 窗，下采样；
  载荷 0/15/30 合并。
- 合并后每类 subsample 至 SAMPLES_PER_CLASS（默认 180 = 6×10×3 齿轮自然样本数）。
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from datasets.data_utils import data_split
from datasets.paths_config import bogie_root

ROOT = str(bogie_root())
NUMPY_DATA_DIR = os.path.join(ROOT, "numpy_data")
BEARING_NORMAL_DIR = os.path.join(
    ROOT, r"转向架轴承数据集\StandardSamples\baseline\with_box"
)
BEARING_FAULTY_DIR = os.path.join(
    ROOT, r"转向架轴承数据集\StandardSamples\fault_sample\with_box"
)
GEAR_DIR = os.path.join(ROOT, r"齿轮故障数据\data")

SPEEDS: Tuple[int, ...] = (1000, 1500, 2000)
BEARING_LOADS: Tuple[int, ...] = (0, 20, 40)
GEAR_LOADS: Tuple[int, ...] = (0, 15, 30)

BEARING_H5_TEMPLATE = "BogieBearing_rpm{speed}.h5"
GEAR_H5_TEMPLATE = "BogieGear_rpm{speed}.h5"
NPY_TEMPLATE = "Bogie_rpm{speed}.npy"

CHANNELS: Tuple[str, ...] = ("FSx", "FSy", "FSz", "NSx", "NSy", "NSz")
NUM_POINTS = 2048 * 11
WIN_SIZE = 1000

BEARING_FILES_PER_LOAD = 3
BEARING_SAMPLES_PER_FILE = 20
GEAR_FILES_PER_LOAD = 6
GEAR_SAMPLES_PER_FILE = 10

SAMPLES_PER_CLASS = (
    GEAR_FILES_PER_LOAD * GEAR_SAMPLES_PER_FILE * len(GEAR_LOADS)
)  # 180

# 13 细分类 -> (目录, 文件名模板)
BEARING_RAW_SPECS: "OrderedDict[str, Tuple[str, str]]" = OrderedDict(
    [
        ("bear_norm", (BEARING_NORMAL_DIR, "baseline_{speed}_{load}_{idx}.csv")),
        ("bear_cage_crack", (BEARING_FAULTY_DIR, "cage_crack_{speed}_{load}_{idx}.csv")),
        (
            "bear_outer_crack_l",
            (BEARING_FAULTY_DIR, "outer_crack_0.5mm_Centered_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_outer_crack_m",
            (BEARING_FAULTY_DIR, "outer_crack_1mm_Centered_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_outer_crack_h",
            (BEARING_FAULTY_DIR, "outer_crack_2mm_Centered_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_outer_pitt_l",
            (BEARING_FAULTY_DIR, "outer_pitting_light_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_outer_pitt_m",
            (BEARING_FAULTY_DIR, "outer_pitting_moderate_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_outer_pitt_h",
            (BEARING_FAULTY_DIR, "outer_pitting_severe_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_roller_crack_l",
            (BEARING_FAULTY_DIR, "roller_crack_0.4mm_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_roller_crack_m",
            (BEARING_FAULTY_DIR, "roller_crack_0.8mm_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_roller_crack_h",
            (BEARING_FAULTY_DIR, "roller_crack_1.2mm_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_roller_pitt_l",
            (BEARING_FAULTY_DIR, "roller_pitting_light_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_roller_pitt_m",
            (BEARING_FAULTY_DIR, "roller_pitting_moderate_{speed}_{load}_{idx}.csv"),
        ),
        (
            "bear_roller_pitt_h",
            (BEARING_FAULTY_DIR, "roller_pitting_severe_{speed}_{load}_{idx}.csv"),
        ),
    ]
)

BEARING_CLASS_GROUPS: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict(
    [
        ("bear_norm", ("bear_norm",)),
        ("bear_cage_crack", ("bear_cage_crack",)),
        (
            "bear_outer_crack",
            ("bear_outer_crack_l", "bear_outer_crack_m", "bear_outer_crack_h"),
        ),
        (
            "bear_outer_pitt",
            ("bear_outer_pitt_l", "bear_outer_pitt_m", "bear_outer_pitt_h"),
        ),
        (
            "bear_roller_crack",
            ("bear_roller_crack_l", "bear_roller_crack_m", "bear_roller_crack_h"),
        ),
        (
            "bear_roller_pitt",
            ("bear_roller_pitt_l", "bear_roller_pitt_m", "bear_roller_pitt_h"),
        ),
    ]
)

GEAR_RAW_SPECS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("gear_crac", "sc_crac_{speed}_{load}_{idx}.csv"),
        ("gear_lack", "sc_lack_{speed}_{load}_{idx}.csv"),
        ("gear_pitt", "sc_pitt_{speed}_{load}_{idx}.csv"),
        ("gear_scor", "sc_scor_{speed}_{load}_{idx}.csv"),
    ]
)

GEAR_CLASS_ORDER: Tuple[str, ...] = tuple(GEAR_RAW_SPECS.keys())


def _h5_path(template: str, speed: int) -> str:
    if speed not in SPEEDS:
        raise ValueError(f"speed must be in {SPEEDS}, got {speed}")
    return os.path.join(NUMPY_DATA_DIR, template.format(speed=speed))


def _fixed_csv_paths(
    directory: str,
    pattern: str,
    speed: int,
    load: int,
    n_files: int,
) -> List[str]:
    paths: List[str] = []
    for idx in range(n_files):
        path = os.path.join(directory, pattern.format(speed=speed, load=load, idx=idx))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def get_bearing_raw_paths(speed: int, load: int) -> "OrderedDict[str, List[str]]":
    out: "OrderedDict[str, List[str]]" = OrderedDict()
    for cls, (directory, pat) in BEARING_RAW_SPECS.items():
        out[cls] = _fixed_csv_paths(
            directory, pat, speed, load, BEARING_FILES_PER_LOAD
        )
    return out


def get_gear_paths(speed: int, load: int) -> "OrderedDict[str, List[str]]":
    out: "OrderedDict[str, List[str]]" = OrderedDict()
    for cls, pat in GEAR_RAW_SPECS.items():
        out[cls] = _fixed_csv_paths(GEAR_DIR, pat, speed, load, GEAR_FILES_PER_LOAD)
    return out


def get_csv_timeseries(
    data_path: Sequence[str],
    num_samples_each_file: int,
    downsampling: bool,
) -> np.ndarray:
    chunks_out: List[np.ndarray] = []
    for p in data_path:
        chunks = pd.read_csv(
            p, sep=r"\s+|,", header=0, chunksize=2048, iterator=True, engine="python"
        )
        data = chunks.get_chunk(NUM_POINTS)
        data = data[list(CHANNELS)].values
        windows = data_split(
            data,
            2048,
            num_samples_each_file,
            win_size=WIN_SIZE,
            downsampling=downsampling,
        )
        chunks_out.append(windows)
    return np.concatenate(chunks_out, axis=0).astype(np.float32)


def load_bearing_raw_from_csv(speed: int) -> "OrderedDict[str, np.ndarray]":
    raw: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for load in BEARING_LOADS:
        paths_map = get_bearing_raw_paths(speed, load)
        for cls, csv_list in paths_map.items():
            arr = get_csv_timeseries(
                csv_list, BEARING_SAMPLES_PER_FILE, downsampling=False
            )
            raw[cls] = np.concatenate([raw[cls], arr], axis=0) if cls in raw else arr
    return raw


def load_gear_from_csv(speed: int) -> "OrderedDict[str, np.ndarray]":
    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for load in GEAR_LOADS:
        paths_map = get_gear_paths(speed, load)
        for cls, csv_list in paths_map.items():
            arr = get_csv_timeseries(
                csv_list, GEAR_SAMPLES_PER_FILE, downsampling=True
            )
            out[cls] = np.concatenate([out[cls], arr], axis=0) if cls in out else arr
    return out


def _concat_keys(
    raw: "OrderedDict[str, np.ndarray]", keys: Sequence[str]
) -> np.ndarray:
    parts = [raw[k] for k in keys if k in raw]
    if len(parts) != len(keys):
        missing = [k for k in keys if k not in raw]
        raise KeyError(f"Missing keys: {missing}")
    return np.concatenate(parts, axis=0)


def merge_bearing_classes(
    raw: "OrderedDict[str, np.ndarray]",
) -> "OrderedDict[str, np.ndarray]":
    merged: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for target, source_keys in BEARING_CLASS_GROUPS.items():
        merged[target] = _concat_keys(raw, source_keys)
    return merged


def equalize_samples(
    arr: np.ndarray,
    n: int = SAMPLES_PER_CLASS,
    seed: int = 0,
) -> np.ndarray:
    """Subsample 或报错：保证每类恰好 n 个样本。"""
    if arr.shape[0] < n:
        raise ValueError(
            f"Need at least {n} samples, got {arr.shape[0]}. "
            "Check CSV files or reduce SAMPLES_PER_CLASS."
        )
    if arr.shape[0] == n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    idx.sort()
    return arr[idx]


def equalize_class_dict(
    class_arrays: "OrderedDict[str, np.ndarray]",
    n: int = SAMPLES_PER_CLASS,
    seed: int = 0,
) -> "OrderedDict[str, np.ndarray]":
    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for i, (name, arr) in enumerate(class_arrays.items()):
        out[name] = equalize_samples(arr, n=n, seed=seed + i)
    return out


def build_bearing_arrays(
    speed: int, equalize: bool = True
) -> "OrderedDict[str, np.ndarray]":
    raw = load_bearing_raw_from_csv(speed)
    merged = merge_bearing_classes(raw)
    if equalize:
        merged = equalize_class_dict(merged)
    return merged


def build_gear_arrays(speed: int, equalize: bool = True) -> "OrderedDict[str, np.ndarray]":
    gear = load_gear_from_csv(speed)
    if equalize:
        gear = equalize_class_dict(gear)
    return gear


def _write_h5(path: str, class_arrays: "OrderedDict[str, np.ndarray]", attrs: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        dg = f.create_group("data")
        for name, arr in class_arrays.items():
            dg.create_dataset(name, data=arr.astype(np.float32), compression="gzip")


def save_h5_bearing(speed: int, overwrite: bool = False) -> str:
    path = _h5_path(BEARING_H5_TEMPLATE, speed)
    if os.path.exists(path) and not overwrite:
        print(f"Skip existing: {path}")
        return path
    arrays = build_bearing_arrays(speed, equalize=True)
    attrs = {
        "speed": speed,
        "loads": list(BEARING_LOADS),
        "samples_per_class": SAMPLES_PER_CLASS,
        "files_per_load": BEARING_FILES_PER_LOAD,
        "samples_per_file": BEARING_SAMPLES_PER_FILE,
        "downsampling": False,
    }
    _write_h5(path, arrays, attrs)
    _print_class_shapes("BogieBearing", speed, arrays)
    return path


def save_h5_gear(speed: int, overwrite: bool = False) -> str:
    path = _h5_path(GEAR_H5_TEMPLATE, speed)
    if os.path.exists(path) and not overwrite:
        print(f"Skip existing: {path}")
        return path
    arrays = build_gear_arrays(speed, equalize=True)
    attrs = {
        "speed": speed,
        "loads": list(GEAR_LOADS),
        "samples_per_class": SAMPLES_PER_CLASS,
        "files_per_load": GEAR_FILES_PER_LOAD,
        "samples_per_file": GEAR_SAMPLES_PER_FILE,
        "downsampling": True,
    }
    _write_h5(path, arrays, attrs)
    _print_class_shapes("BogieGear", speed, arrays)
    return path


def save_all_h5(overwrite: bool = False) -> None:
    for speed in SPEEDS:
        save_h5_bearing(speed, overwrite=overwrite)
        save_h5_gear(speed, overwrite=overwrite)


def _class_arrays_to_xy(
    class_arrays: "OrderedDict[str, np.ndarray]",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    xs, ys, names, counts = [], [], [], []
    for label, (name, arr) in enumerate(class_arrays.items()):
        xs.append(arr)
        ys.append(np.full(arr.shape[0], label, dtype=np.int32))
        names.append(name)
        counts.append(int(arr.shape[0]))
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int32)
    return x, y, names, counts


def load_rpm_arrays(
    speed: int,
    include_bearing: bool = True,
    include_gear: bool = True,
    equalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    class_arrays: "OrderedDict[str, np.ndarray]" = OrderedDict()
    meta: Dict[str, object] = {
        "speed": speed,
        "samples_per_class": SAMPLES_PER_CLASS,
        "protocol": {
            "bearing": {
                "loads": list(BEARING_LOADS),
                "files_per_load": BEARING_FILES_PER_LOAD,
                "samples_per_file": BEARING_SAMPLES_PER_FILE,
                "downsampling": False,
            },
            "gear": {
                "loads": list(GEAR_LOADS),
                "files_per_load": GEAR_FILES_PER_LOAD,
                "samples_per_file": GEAR_SAMPLES_PER_FILE,
                "downsampling": True,
            },
        },
    }

    if include_bearing:
        class_arrays.update(build_bearing_arrays(speed, equalize=equalize))
    if include_gear:
        for k in GEAR_CLASS_ORDER:
            class_arrays[k] = build_gear_arrays(speed, equalize=equalize)[k]

    x, y, class_names, samples_per_class = _class_arrays_to_xy(class_arrays)
    meta["class_names"] = class_names
    meta["samples_per_class_list"] = samples_per_class
    meta["num_classes"] = len(class_names)
    meta["num_samples"] = int(x.shape[0])
    return x, y, class_names, meta


def export_rpm_npy(
    speed: int,
    overwrite: bool = False,
    include_bearing: bool = True,
    include_gear: bool = True,
) -> str:
    os.makedirs(NUMPY_DATA_DIR, exist_ok=True)
    out_path = os.path.join(NUMPY_DATA_DIR, NPY_TEMPLATE.format(speed=speed))
    if os.path.exists(out_path) and not overwrite:
        print(f"Skip existing: {out_path}")
        return out_path

    class_arrays: "OrderedDict[str, np.ndarray]" = OrderedDict()
    if include_bearing:
        class_arrays.update(build_bearing_arrays(speed, equalize=True))
    if include_gear:
        class_arrays.update(build_gear_arrays(speed, equalize=True))
    x, y, class_names, counts = _class_arrays_to_xy(class_arrays)
    meta = {
        "speed": speed,
        "samples_per_class": SAMPLES_PER_CLASS,
        "samples_per_class_list": counts,
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_samples": int(x.shape[0]),
    }
    np.save(
        out_path,
        {"x": x, "y": y, "rpm": speed, "class_names": class_names, "meta": meta},
        allow_pickle=True,
    )
    _print_class_shapes("Bogie_npy", speed, class_arrays)
    print(f"Saved {out_path}: x={x.shape}")
    return out_path


def rebuild_all(overwrite: bool = False) -> None:
    save_all_h5(overwrite=overwrite)
    for speed in SPEEDS:
        export_rpm_npy(speed, overwrite=overwrite)


def list_bogie_h5_files() -> List[str]:
    if not os.path.isdir(NUMPY_DATA_DIR):
        return []
    return sorted(
        os.path.join(NUMPY_DATA_DIR, fn)
        for fn in os.listdir(NUMPY_DATA_DIR)
        if fn.endswith(".h5") and fn.startswith("Bogie")
    )


def inspect_h5(path: str) -> Dict[str, object]:
    with h5py.File(path, "r") as f:
        dg = f["data"]
        shapes = {k: tuple(dg[k].shape) for k in dg.keys()}
        attrs = dict(f.attrs)
    return {"path": path, "attrs": attrs, "shapes": shapes}


def _print_class_shapes(tag: str, speed: int, arrays: "OrderedDict[str, np.ndarray]") -> None:
    print(f"[{tag} rpm{speed}]")
    for name, arr in arrays.items():
        print(f"  {name}: {arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bogie CSV -> h5 / npy rebuild")
    parser.add_argument("--inspect", action="store_true", help="Inspect h5 files")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all h5 and npy")
    parser.add_argument("--rebuild-h5", action="store_true", help="Rebuild h5 only")
    parser.add_argument("--export", action="store_true", help="Export npy only")
    parser.add_argument("--speed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        for path in list_bogie_h5_files():
            info = inspect_h5(path)
            print("=" * 60, os.path.basename(info["path"]))
            print("  attrs:", info["attrs"])
            for k, sh in info["shapes"].items():
                print(f"    {k}: {sh}")
        return

    if args.rebuild:
        rebuild_all(overwrite=args.overwrite)
        return

    if args.rebuild_h5:
        speeds = [args.speed] if args.speed else list(SPEEDS)
        for s in speeds:
            save_h5_bearing(s, overwrite=args.overwrite)
            save_h5_gear(s, overwrite=args.overwrite)
        return

    if args.export:
        speeds = [args.speed] if args.speed else list(SPEEDS)
        for s in speeds:
            export_rpm_npy(s, overwrite=args.overwrite)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
