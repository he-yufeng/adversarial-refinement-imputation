# 对抗式精修何时有用？—— R3GAN 用于时间序列填补

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 中文

本仓库是以下论文的代码与论文配套仓：

> **When Does Adversarial Refinement Help? A Negative Result and Open Problem in Adapting R3GAN to Time Series Imputation**
> 何宇峰（香港大学）
> *第 12 届 SIGKDD 时间序列挖掘与学习研讨会（**MiLeTS 2026**）。*
> 📄 论文：[`FMGAN/paper/main_milets2026_sigconf.pdf`](FMGAN/paper/main_milets2026_sigconf.pdf)

---

## 一句话总结

扩散模型与 Transformer 已在多变量时间序列填补上取代 GAN，主要理由是 **GAN 训练不稳定**。[R3GAN](https://github.com/brownvc/R3GAN)（NeurIPS 2024）用正则化相对论损失 + 可证明收敛消除了这种不稳定——于是我们问：*稳定的现代 GAN 能否让对抗式填补复活？*

我们把 R3GAN 适配到一维时序（**R3GAN-1D**），做成带频域判别器的「粗到精」精修器，在 3 个数据集、15+ 配置上系统实验。**答案是一个界定清晰的负结果：**

- ✅ R3GAN-1D 对**分布上不合理**的粗填补提升巨大——相对 zero/mean fill **MAE 降低 48–70%**。
- ❌ 但对**强粗填补**（线性插值）**无提升**（|Δ| < 1.5%），作为独立填补器**比 BRITS 差 5.8 倍**。
- 🔍 教科书式解释「GAN 优化分布目标、非逐点目标」**站不住**——扩散模型同样是分布式目标，却在这个逐点任务上 SoTA。我们的消融把问题指向**对抗信号本身**，并把确切机制列为**开放问题**。

这是一篇负结果 workshop 论文：价值在于**控制掉了 GAN 不稳定这一混淆变量**，把「为何对抗判别无法提供有用的条件精修梯度、而去噪/扩散目标可以」这个问题问得更精确。

## 核心结果——精修只救得动弱粗填补

MAE ↓ 越低越好；`Δ` = 相对 MAE 降幅（正=改进）。

| 数据集 | 粗填补方法 | 精修前 | 精修后 | Δ |
|---|---|:---:|:---:|:---:|
| Weather | Zero fill | 0.728 | 0.228 | **+68.6%** |
| | Mean fill | 0.728 | 0.223 | **+69.4%** |
| | 线性插值 | 0.067 | 0.067 | +1.1% |
| Electricity | Zero fill | 0.832 | 0.427 | **+48.7%** |
| | Mean fill | 0.831 | 0.429 | **+48.4%** |
| | 线性插值 | 0.164 | 0.165 | −0.7% |
| AirQuality | Zero fill | 0.765 | 0.228 | **+70.2%** |
| | 线性插值 | 0.151 | 0.152 | −0.4% |

与成熟方法对比（Weather，25% 点缺失）：BRITS **0.039** / SAITS 0.062 / 线性插值 0.067 / R3GAN-1D+线性 0.067 / R3GAN-1D 独立 0.228。

> 那些大幅百分比提升，都是相对没人会真用的 *trivial* baseline；R3GAN-1D 从未超过一个像样的 baseline。

## 复现论文表格

逐次实验的原始输出在 [`FMGAN/results/`](FMGAN/results/)，上表由一个脚本一键重建（无需 GPU、无需训练）：

```bash
python3 FMGAN/analysis.py
```

## 仓库结构

论文代码在 [`FMGAN/`](FMGAN/)（**F**oundation-**M**odel 粗填补 + **GAN** 精修）：`models/`（R3GAN-1D 架构 + 频域判别器）、`train_refiner.py`（粗到精训练）、`foundation_model/`（MOMENT 包装）、`evaluation/`（指标 + BRITS/SAITS/CSDI baseline）、`data/`（统一加载，含 point/block/subsequence 缺失）、`scripts/`、`results/`（原始 results.json）、`analysis.py`、`paper/`（MiLeTS 2026 camera-ready）。

> 早期硕士论文模型（R3GAN 图像式 MTSI）及其 SSGAN/TimesNet baseline 与 Dash GUI **不属于本论文**，已保留在 [`thesis-archive`](https://github.com/he-yufeng/r3gan-time-series-imputation/tree/thesis-archive) 分支。

## 安装

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

复现表格（`analysis.py`）仅需 Python 标准库；重跑实验需完整栈（PyTorch、PyPOTS、MOMENT）+ GPU，见 [`FMGAN/requirements.txt`](FMGAN/requirements.txt)。

## 局限与开放问题

- 最直接的对照——**recon-only（移除判别器）消融**——是最有价值的下一个实验；当前证据（重建权重 sweep）是间接的。
- **协议内扩散 baseline**（CSDI/FGTI 跑全 3 数据集）、**多 seed 误差棒**、**block/高缺失率/MNAR** 会进一步加强结论。
- 开放问题：*为何学到的扩散去噪器能做好逐点填补，而 GAN 判别器不能，尽管两者都优化分布目标？* 我们猜测缺陷在于**学习信号的形式**（全局判别 vs 条件式逐点回归），而非分布匹配本身。

## 引用

见 [README.md](README.md) 的 BibTeX。

## 联系

何宇峰 — [@he-yufeng](https://github.com/he-yufeng) · he-yufeng@connect.hku.hk
