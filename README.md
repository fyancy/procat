# ProCAT

Reproducible **test-time adaptation (TTA)** benchmark on three 1D bearing fault-diagnosis datasets:

- **SQ** — eight tasks (T1–T8): cross-speed domain shift, dynamic noise, Dirichlet label shift
- **Bogie** — six tasks (T1–T6): cross-RPM domain shift with varying Dirichlet γ
- **OUB** — eight tasks (T1–T8): cross-domain Ottawa University Bearing (2048-point, `light_trial1` subset)

Experiments use the **online-batch** protocol (batch size 64, seed 0) and compare **10 baseline methods**:

`source_only`, `bn_adapt`, `rotta`, `cotta`, `petta`, `tea`, `tact`, `eata`, `tribe_official`, `tribe`

Pre-trained **TFN** and **ResNet18** source checkpoints are included under `checkpoints/`. Raw dataset files are **not** distributed.

## Quick start

```bash
pip install -r requirements.txt
cp configs/paths.example.yaml configs/paths.yaml   # edit paths to your data caches
```

Requires Python 3.10+ and PyTorch (CUDA recommended).

Set `TTA_DATA_ROOT` to the parent folder containing `SQdata/`, `OUBdata/`, etc., or edit `configs/paths.yaml`.

## Run experiments

```bash
# SQ — all 8 tasks, TFN
python -m tta.sq.run_sq_full eval --tasks all --model tfn --device cuda

# Bogie — all 6 tasks
python -m tta.bogie.run_bogie_full eval --tasks all --model tfn --device cuda

# OUB light_trial1 — all 8 tasks
python -m tta.oub.run_oub_2048_full eval --subset light_trial1 --tasks all --model tfn --device cuda
```

## Data files (not included)

| Dataset | Required preprocessed file(s) |
|---|---|
| SQ | `SQdata/numpy_data_resampled/sq_no_noise_resampled_for_att.npy` |
| Bogie | `转向架齿轮轴承/numpy_data/Bogie_rpm{1000,1500,2000}.npy` |
| OUB | `OUBdata/numpy_resampled/oub_len2048_nonoverlap_trialwise.npy` |

## License

See LICENSE (to be added).
