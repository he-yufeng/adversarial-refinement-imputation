# When Does Adversarial Refinement Help? — R3GAN for Time Series Imputation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/he-yufeng/adversarial-refinement-imputation/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/adversarial-refinement-imputation/actions/workflows/ci.yml)

[中文文档](README_CN.md) | English

Code and paper for:

> **When Does Adversarial Refinement Help? A Negative Result and Open Problem in Adapting R3GAN to Time Series Imputation**
> Yufeng He (The University of Hong Kong)
> *12th SIGKDD Workshop on Mining and Learning from Time Series (**MiLeTS 2026**).*
> 📄 Paper: [`FMGAN/paper/main_milets2026_sigconf.pdf`](FMGAN/paper/main_milets2026_sigconf.pdf)

---

## TL;DR

Diffusion models and transformers have supplanted GANs for multivariate time series imputation, largely on grounds of **GAN training instability**. [R3GAN](https://github.com/brownvc/R3GAN) (NeurIPS 2024) removes that instability with regularized relativistic losses and provable convergence — so we asked: *does a stable, modern GAN revive adversarial imputation?*

We adapt R3GAN to 1D temporal data (**R3GAN-1D**) as a coarse-to-fine **refiner** with a frequency-domain discriminator, and run a systematic study across 3 datasets and 15+ configurations. **The answer is a clearly-scoped negative result:**

- ✅ R3GAN-1D **dramatically improves distributionally implausible** coarse imputers — **48–70% MAE reduction** over zero / mean fill.
- ❌ It yields **no improvement over a strong coarse imputer** (linear interpolation, |Δ| < 1.5%), and as a standalone imputer it is **5.8× worse than BRITS**.
- 🔍 The textbook "GANs optimize distributional, not point-wise, objectives" explanation **cannot be the whole story** — diffusion models are *also* distributional yet are SoTA at this point-wise task. Our ablations instead implicate the **adversarial signal specifically**, and we frame the precise mechanism as an **open problem**.

This is a negative-result workshop paper: the value is in **controlling away GAN instability** and sharpening *why* adversarial discrimination fails to provide useful conditional-refinement gradients where denoising/diffusion objectives succeed.

## Key result — refinement only helps weak coarse imputers

MAE ↓ (lower is better). `Δ` = relative MAE reduction (positive = improvement).

| Dataset     | Coarse method     | Before | After | Δ          |
|-------------|-------------------|:------:|:-----:|:----------:|
| Weather     | Zero fill         | 0.728  | 0.228 | **+68.6%** |
|             | Mean fill         | 0.728  | 0.223 | **+69.4%** |
|             | Linear interp     | 0.067  | 0.067 | +1.1%      |
| Electricity | Zero fill         | 0.832  | 0.427 | **+48.7%** |
|             | Mean fill         | 0.831  | 0.429 | **+48.4%** |
|             | Linear interp     | 0.164  | 0.165 | −0.7%      |
| AirQuality  | Zero fill         | 0.765  | 0.228 | **+70.2%** |
|             | Linear interp     | 0.151  | 0.152 | −0.4%      |

Standalone vs. established methods (Weather, 25% point-missing):

| Method               | Type         | MAE ↓ |
|----------------------|--------------|:-----:|
| BRITS                | RNN          | **0.039** |
| SAITS                | Transformer  | 0.062 |
| Linear interpolation | Simple       | 0.067 |
| R3GAN-1D + linear    | GAN refine   | 0.067 |
| R3GAN-1D standalone  | GAN          | 0.228 |

> The large percentage gains are improvements over *trivial* baselines no practitioner would deploy. R3GAN-1D never improves on a competent baseline.

## Reproduce the paper tables

The raw per-run outputs live in [`FMGAN/results/`](FMGAN/results/); the tables above are regenerated from them by a single script (no GPU, no training):

```bash
python3 FMGAN/analysis.py
```

This walks `FMGAN/results/results/phase1_*/results.json` + `baseline_*.json` and prints the refinement and comparison tables. It prints **every** run, so you also see the spread behind each cell: for strong (linear-interpolation) coarse fills, all variants land in **[−21.9%, +1.1%]** (mean −3.1%), reinforcing that adversarial refinement never meaningfully improves a competent baseline — and can hurt it.

## Repository structure

This repo is the companion to the MiLeTS 2026 paper. The study code lives under [`FMGAN/`](FMGAN/) (**F**oundation-**M**odel-coarse + **GAN**-refiner):

```
.
├── FMGAN/
│   ├── models/r3gan_1d.py     # R3GAN-1D architecture (1D adaptation + frequency-domain discriminator)
│   ├── train_refiner.py       # coarse-to-fine refinement training
│   ├── foundation_model/      # MOMENT wrapper (a foundation-model coarse imputer)
│   ├── evaluation/            # metrics + BRITS / SAITS / CSDI baselines (PyPOTS)
│   ├── data/                  # unified loaders (point / block / subsequence missingness)
│   ├── scripts/               # experiment runners
│   ├── configs/               # default config
│   ├── results/               # raw results.json (paper tables reproduce from these)
│   ├── analysis.py            # reproduce paper tables
│   └── paper/                 # MiLeTS 2026 camera-ready (LaTeX + PDF), references, figures
├── requirements.txt
└── LICENSE                    # MIT
```

> The author's earlier undergraduate final-year project (FYP) — an R3GAN adaptation to image-style MTSI, plus SSGAN / TimesNet baselines and a Dash GUI — is **not** part of this paper; it is preserved on the [`fyp-archive`](https://github.com/he-yufeng/adversarial-refinement-imputation/tree/fyp-archive) branch.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Reproducing the tables (`analysis.py`) needs only Python stdlib + the bundled JSON. Re-running experiments needs the full stack (PyTorch, PyPOTS, MOMENT) and a GPU — see [`FMGAN/requirements.txt`](FMGAN/requirements.txt) and [`FMGAN/scripts/`](FMGAN/scripts/).

## Datasets

Standard public benchmarks, evaluated at 25% point-missing (MCAR):

- **Weather** — 52K timesteps, 21 meteorological features
- **Electricity** — 140K timesteps, 370 client-consumption features
- **AirQuality** — 8.7K timesteps, 36 PM2.5 stations (13% originally missing)

They are downloaded on demand via the [PyPOTS](https://github.com/WenjieDu/PyPOTS) / `tsdb` ecosystem (see `FMGAN/data/`); none are bundled.

## Limitations & open problem

Honest scope (also in the paper, expanded for camera-ready):

- The most direct control — a **reconstruction-only (discriminator-removed) ablation** — is the single most valuable next experiment; our current evidence (a reconstruction-weight sweep) is indirect.
- An **in-protocol diffusion baseline** (CSDI / FGTI on all three datasets), **multi-seed error bars**, and **block / higher-rate / MNAR** settings would further strengthen the claim.
- The open problem: *why does a learned diffusion denoiser succeed at point-wise imputation while a GAN discriminator fails, when both optimize distributional objectives?* We conjecture the deficit is in the **form of the learning signal** (global discrimination vs. conditional per-position regression), not in distribution-matching per se.

## Citation

```bibtex
@inproceedings{he2026adversarial,
  title     = {When Does Adversarial Refinement Help? A Negative Result and Open
               Problem in Adapting R3GAN to Time Series Imputation},
  author    = {He, Yufeng},
  booktitle = {12th SIGKDD Workshop on Mining and Learning from Time Series (MiLeTS)},
  year      = {2026}
}
```

## Acknowledgments & references

- **R3GAN** — Huang, Gokaslan, Kuleshov, Tompkin. *The GAN is dead; long live the GAN! A Modern GAN Baseline.* NeurIPS 2024.
- **BRITS** — Cao et al. NeurIPS 2018 · **SAITS** — Du et al. 2023 · **CSDI** — Tashiro et al. NeurIPS 2021.
- Baselines run via [PyPOTS](https://github.com/WenjieDu/PyPOTS); coarse foundation-model imputer via [MOMENT](https://github.com/moment-timeseries-foundation-model/moment).

## License

Original code is released under the [MIT License](LICENSE). The R3GAN-1D implementation is an original 1D adaptation; see the references above for the upstream ideas it builds on.

## Contact

Yufeng He — [@he-yufeng](https://github.com/he-yufeng) · he-yufeng@connect.hku.hk
