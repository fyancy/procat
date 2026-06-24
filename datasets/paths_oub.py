"""Path helpers for Ottawa UOB (v43hmbwxpm v2) bearing dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from datasets.paths_config import oub_cache_dir, oub_raw_dir

DEFAULT_RAW_DIR = oub_raw_dir()
DEFAULT_CACHE_DIR = oub_cache_dir()

# Folder names in v43hmbwxpm-2.zip (label 0..4)
FAULT_FOLDERS: Tuple[str, ...] = (
    "1 Data collected from a healthy bearing",
    "2 Data collected from a bearing with inner race fault",
    "3 Data collected from a bearing with outer race fault",
    "4 Data collected from a bearing with ball fault",
    "5 Data collected from a bearing with a combination of faults",
)

# File prefix per label: H/I/O/B/C
CLASS_PREFIX: Tuple[str, ...] = ("H", "I", "O", "B", "C")

# Speed profile codes in filename: A/B/C/D -> domain 0..3 (B1..B4)
DOMAIN_CODES: Tuple[str, ...] = ("A", "B", "C", "D")
DOMAIN_NAMES: Tuple[str, ...] = ("B1_inc", "B2_dec", "B3_inc_dec", "B4_dec_inc")

# Trials 1..3 in filenames
TRIAL_IDS: Tuple[int, ...] = (1, 2, 3)


def get_oub_mat_path(raw_dir: Path, label: int, domain: int, trial: int) -> Path:
    """Return path to one .mat file: e.g. H-A-1.mat for healthy, increasing, trial 1."""
    if not (0 <= label < len(CLASS_PREFIX)):
        raise ValueError(f"label out of range: {label}")
    if not (0 <= domain < len(DOMAIN_CODES)):
        raise ValueError(f"domain out of range: {domain}")
    if trial not in TRIAL_IDS:
        raise ValueError(f"trial must be in {TRIAL_IDS}, got {trial}")
    folder = raw_dir / FAULT_FOLDERS[label]
    fname = f"{CLASS_PREFIX[label]}-{DOMAIN_CODES[domain]}-{trial}.mat"
    return folder / fname


def list_all_mat_files(raw_dir: Path | None = None) -> List[Dict[str, object]]:
    """Enumerate all 60 canonical mat files with parsed metadata."""
    root = DEFAULT_RAW_DIR if raw_dir is None else Path(raw_dir)
    entries: List[Dict[str, object]] = []
    for label, folder_name in enumerate(FAULT_FOLDERS):
        for domain, code in enumerate(DOMAIN_CODES):
            for trial in TRIAL_IDS:
                path = root / folder_name / f"{CLASS_PREFIX[label]}-{code}-{trial}.mat"
                entries.append(
                    {
                        "path": str(path),
                        "label": label,
                        "domain": domain,
                        "trial": trial - 1,
                        "stem": path.stem,
                        "folder": folder_name,
                    }
                )
    return entries
