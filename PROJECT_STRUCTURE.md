# MTSIR3-GAN Project Structure

Overview of the directory layout and where the important files live. Each
top-level folder also has its own `README.md` with usage and provenance.

## Directory Tree

```
MTSIR3-GAN/
├── README.md / README_CN.md          # Main documentation (EN / 中文)
├── USAGE_GUIDE.md                    # Detailed usage instructions
├── PROJECT_STRUCTURE.md              # This file
├── requirements.txt                  # Core dependencies
├── .gitignore / .gitattributes       # Git config (ignores, linguist hints)
├── LICENSE                           # MIT (original code only)
│
├── R3GAN/                            # Thesis model: R3GAN adapted to MTSI
│   ├── train.py                      # Training entry point (per-dataset presets)
│   ├── gen_timeseries.py             # Generation / imputation
│   ├── calc_metrics.py               # MAE / MSE / RMSE evaluation
│   ├── dataset_tool.py               # Dataset preparation
│   ├── process_air_quality.py        # AirQuality preprocessing
│   ├── R3GAN/                        # Networks, resamplers, fused ops, trainer
│   ├── training/                     # Training loop, losses (RpGAN+R1+R2), augment
│   ├── torch_utils/, dnnlib/, metrics/, legacy.py   # StyleGAN3 infrastructure
│   └── README.md                     # ⚠️ provenance: NVIDIA StyleGAN3 + R3GAN
│
├── SSGAN/                            # Baseline: Miao et al., AAAI 2021
│   ├── main.py, data_loader.py, preprocess.py, utils.py
│   ├── models/                       # Based_on_BRITS, brits, rits, gru_d, ...
│   └── README.md
│
├── TimesNet/                        # Baseline: vendored from THUML TSLib (MIT)
│   ├── run.py                        # Entry (--task_name imputation)
│   ├── models/                       # TimesNet + ~30 other architectures
│   ├── layers/, data_provider/, exp/, utils/
│   └── README.md
│
├── PURE-GUIv2.0/                    # Dash web interface
│   ├── app_dad.py                    # Dash app entry point
│   ├── pages/                        # data_analysis_dad, time_imputation_dad,
│   │                                 #   model_visualization
│   ├── model_files/  model_results/  uploaded_files/   # not bundled (gitignored)
│   └── README.md
│
├── FMGAN/                           # Extended empirical study (coarse-to-fine)
│   ├── train_refiner.py              # R3GAN-1D refiner training (--device cuda|mps|cpu)
│   ├── models/r3gan_1d.py            # 1D R3GAN architecture
│   ├── foundation_model/             # MOMENT wrapper
│   ├── evaluation/                   # Metrics + baselines (SAITS/BRITS/CSDI)
│   ├── data/                         # Unified loaders (point/block/subsequence)
│   ├── scripts/                      # Experiment runners
│   ├── configs/, requirements.txt    # Study config + extra deps
│   └── results/                      # Run outputs (gitignored)
│
└── datasets/                        # Sample data + download pointers
    ├── AirQuality/  (pm25_ground.txt, pm25_missing.txt)
    ├── PSM/         (test.csv, test_label.csv)
    ├── PhysioNet/   (link.txt)
    └── README.md
```

## Key Files

### Training entry points

| Component | File | Example |
|-----------|------|---------|
| MTSIR3-GAN (thesis) | `R3GAN/train.py` | `python train.py --data=<path> --gpus=1 --preset=AirQuality_MTSI` |
| SSGAN baseline | `SSGAN/main.py` | `python main.py --epochs=50` |
| TimesNet baseline | `TimesNet/run.py` | `python run.py --task_name imputation` |
| FMGAN refiner | `FMGAN/train_refiner.py` | `python train_refiner.py --dataset AirQuality --coarse linear` |

### Architecture & losses

| File | Description |
|------|-------------|
| `R3GAN/R3GAN/Networks.py` | Generator & Discriminator |
| `R3GAN/training/loss.py` | RpGAN + R₁ + R₂ loss |
| `FMGAN/models/r3gan_1d.py` | 1D adaptation (refiner G/D) |
| `SSGAN/models/Based_on_BRITS.py` | 3-player GAN (G, D, C) |
| `TimesNet/models/TimesNet.py` | 2D-variation modeling |

## Data Flow (training)

```
Raw data (CSV/TXT)
  → preprocessing (dataset_tool.py / preprocess.py / FMGAN data loaders)
  → Dataset + DataLoader (batching, masking)
  → training (train.py / main.py / run.py / train_refiner.py)
  → checkpoints (.pkl / .pt / .pth)
  → evaluation (calc_metrics.py / evaluation/metrics.py)
```

## Checkpoints

- **R3GAN** (StyleGAN3-style `.pkl`): dict with `G`, `D`, `G_ema`,
  `training_set_kwargs`, `augment_pipe`. Load via
  `R3GAN/legacy.load_network_pkl(...)['G_ema']`.
- **FMGAN refiner** (`.pt`): saved by `train_refiner.py` as `best_model.pt`
  alongside `results.json` and `training_log.json` in the run's `--outdir`.
- **SSGAN / TimesNet** (`.pth`): standard `state_dict` checkpoints.

> Pretrained weights are **not** committed (size). Train locally or supply your
> own; the GUI loads them from `PURE-GUIv2.0/model_files/`.

## R3GAN dataset presets (`R3GAN/train.py`)

- `AirQuality_MTSI` — 36-feature air quality
- `PSM_MTSI` — 25-feature server metrics
- `PhysioNet_MTSI` — 41-feature clinical data

## Notes

- `__pycache__/`, large binaries (`*.pth/*.pt/*.npy`), `datasets/`, and
  `FMGAN/results/` are git-ignored.
- Vendored code (TimesNet, StyleGAN3 infra, paper templates) is marked
  `linguist-vendored`/`-documentation` in `.gitattributes`.
- Device selection auto-detects CUDA → MPS → CPU (FMGAN exposes `--device`).

---

For detailed usage of each component, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
