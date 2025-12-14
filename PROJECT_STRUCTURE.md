# MTSIR3-GAN Project Structure

Complete overview of the project directory structure and file organization.

## Directory Tree

```
MTSIR3-GAN/
│
├── README.md                          # Main documentation (English)
├── README_CN.md                       # Chinese documentation
├── USAGE_GUIDE.md                     # Detailed usage instructions
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
│
├── figures/                           # Visualization assets for reports
│   ├── ganstructure.png
│   ├── r3gan.png
│   ├── stylegan2.png
│   ├── Hyper-*.png                   # Hyperparameter analysis plots
│   ├── violin_plot_*.png             # Distribution visualizations
│   ├── GUI_*.png                     # GUI screenshots
│   └── DatasetAna-*.png              # Dataset analysis figures
│
├── TSImputation-master/               # Core model implementations
│   │
│   ├── datasets/                      # Dataset directory
│   │   ├── AirQuality/
│   │   │   ├── pm25_ground.txt       # Ground truth PM2.5 data
│   │   │   └── pm25_missing.txt      # Data with missing values
│   │   ├── PhysioNet/
│   │   │   └── link.txt              # Download instructions
│   │   └── PSM/
│   │       ├── train.csv             # Training data
│   │       ├── test.csv              # Test data
│   │       └── test_label.csv        # Test labels
│   │
│   ├── R3GAN/                         # MTSIR3-GAN (R3GAN adaptation)
│   │   │
│   │   ├── train.py                  # Main training script
│   │   ├── gen_timeseries.py         # Time series generation script
│   │   ├── calc_metrics.py           # Evaluation metrics computation
│   │   ├── dataset_tool.py           # Dataset preparation utilities
│   │   ├── process_air_quality.py    # AirQuality preprocessing
│   │   ├── legacy.py                 # Model loading utilities
│   │   │
│   │   ├── R3GAN/                    # Core architecture modules
│   │   │   ├── Networks.py           # Generator & Discriminator networks
│   │   │   ├── FusedOperators.py     # Optimized operations
│   │   │   ├── Resamplers.py         # Up/downsampling modules
│   │   │   └── Trainer.py            # Training utilities
│   │   │
│   │   ├── training/                 # Training components
│   │   │   ├── training_loop.py      # Main training loop
│   │   │   ├── loss.py               # Loss functions (RpGAN+R1+R2)
│   │   │   ├── networks.py           # Network definitions
│   │   │   ├── dataset.py            # Dataset class
│   │   │   └── augment.py            # Data augmentation
│   │   │
│   │   ├── torch_utils/              # PyTorch utilities
│   │   │   ├── ops/                  # Custom CUDA operations
│   │   │   │   ├── bias_act.py       # Bias & activation fusion
│   │   │   │   ├── upfirdn2d.py      # Up/downsampling filters
│   │   │   │   ├── conv2d_gradfix.py # Gradient fixes
│   │   │   │   ├── *.cu              # CUDA kernels
│   │   │   │   └── *.cpp             # C++ extensions
│   │   │   ├── custom_ops.py         # Custom operation loader
│   │   │   ├── misc.py               # Miscellaneous utilities
│   │   │   ├── persistence.py        # Model saving/loading
│   │   │   └── training_stats.py     # Training statistics
│   │   │
│   │   ├── metrics/                  # Evaluation metrics
│   │   │   ├── metric_main.py        # Metric computation entry
│   │   │   ├── metric_utils.py       # Metric utilities
│   │   │   ├── frechet_inception_distance.py  # FID score
│   │   │   ├── inception_score.py    # IS score
│   │   │   ├── kernel_inception_distance.py   # KID score
│   │   │   └── precision_recall.py   # Precision/Recall
│   │   │
│   │   └── dnnlib/                   # Deep learning library
│   │       ├── __init__.py
│   │       └── util.py               # General utilities
│   │
│   ├── SSGAN/                         # Semi-Supervised GAN baseline
│   │   │
│   │   ├── main.py                   # Training & evaluation script
│   │   ├── data_loader.py            # Data loading utilities
│   │   ├── preprocess.py             # Data preprocessing
│   │   ├── utils.py                  # Helper functions
│   │   │
│   │   └── models/                   # Model architectures
│   │       ├── Based_on_BRITS.py     # Main SSGAN model
│   │       ├── brits.py              # BRITS imputation model
│   │       ├── brits_i.py            # BRITS components
│   │       ├── rits.py               # RITS model
│   │       ├── rits_i.py             # RITS components
│   │       ├── discriminator.py      # Discriminator network
│   │       ├── discriminator2.py     # Alternative discriminator
│   │       ├── classifier.py         # Classification network
│   │       ├── gru_d.py              # GRU-D baseline
│   │       └── m_rnn.py              # Multi-directional RNN
│   │
│   └── TimesNet/                      # TimesNet baseline
│       │
│       ├── run.py                    # Main entry point
│       │
│       ├── exp/                      # Experiment classes
│       │   ├── exp_basic.py          # Base experiment class
│       │   ├── exp_imputation.py     # Imputation experiment
│       │   ├── exp_long_term_forecasting.py
│       │   ├── exp_short_term_forecasting.py
│       │   ├── exp_anomaly_detection.py
│       │   └── exp_classification.py
│       │
│       ├── models/                   # Model implementations
│       │   ├── TimesNet.py           # Main TimesNet model
│       │   ├── Autoformer.py         # Autoformer baseline
│       │   ├── Transformer.py        # Transformer baseline
│       │   ├── Informer.py           # Informer baseline
│       │   ├── PatchTST.py           # PatchTST model
│       │   ├── iTransformer.py       # iTransformer model
│       │   ├── DLinear.py            # DLinear baseline
│       │   └── [30+ other models]    # Additional baselines
│       │
│       ├── layers/                   # Network layers
│       │   ├── Embed.py              # Embedding layers
│       │   ├── Conv_Blocks.py        # Convolutional blocks
│       │   ├── SelfAttention_Family.py  # Attention mechanisms
│       │   ├── Transformer_EncDec.py    # Transformer encoder/decoder
│       │   ├── AutoCorrelation.py       # Auto-correlation layer
│       │   └── [10+ other layers]
│       │
│       ├── data_provider/            # Data loading
│       │   ├── data_factory.py       # Dataset factory
│       │   ├── data_loader.py        # Data loaders
│       │   ├── m4.py                 # M4 dataset
│       │   └── uea.py                # UEA datasets
│       │
│       └── utils/                    # Utility functions
│           ├── tools.py              # General tools
│           ├── metrics.py            # Evaluation metrics
│           ├── losses.py             # Loss functions
│           ├── masking.py            # Masking utilities
│           ├── timefeatures.py       # Time feature encoding
│           └── [5+ other utilities]
│
├── PURE-GUIv2/                        # Latest web interface
│   │
│   ├── app_dad.py                    # Main Dash application
│   │
│   ├── pages/                        # UI module pages
│   │   ├── data_analysis_dad.py      # Data analysis interface
│   │   ├── time_imputation_dad.py    # Imputation interface
│   │   └── model_visualization.py    # Visualization interface
│   │
│   ├── model_files/                  # Pretrained model checkpoints
│   │   ├── AirQuality_R3GAN.pth
│   │   ├── AirQuality_SSGAN.pth
│   │   └── AirQuality_TimesNet.pth
│   │
│   ├── model_results/                # Model prediction results
│   │   ├── R3GAN/
│   │   │   ├── pred.npy
│   │   │   └── true.npy
│   │   ├── SSGAN/
│   │   │   ├── pred.npy
│   │   │   └── true.npy
│   │   └── TimesNet/
│   │       ├── pred.npy
│   │       └── true.npy
│   │
│   └── uploaded_files/               # User-uploaded data (gitignored)
│       └── [user CSV files]
│
└── PURE-GUI/                          # Previous GUI version (reference)
    └── [similar structure to GUIv2]

```

## Key Files Explained

### Core Training Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `TSImputation-master/R3GAN/train.py` | Train MTSIR3-GAN | `python train.py --data=<path> --gpus=1` |
| `TSImputation-master/SSGAN/main.py` | Train SSGAN | `python main.py --epochs=50` |
| `TSImputation-master/TimesNet/run.py` | Train TimesNet | `python run.py --task_name imputation` |

### Model Architecture Files

| File | Description |
|------|-------------|
| `R3GAN/R3GAN/Networks.py` | Generator & Discriminator definitions |
| `R3GAN/training/loss.py` | RpGAN + R1 + R2 loss implementation |
| `R3GAN/training/networks.py` | ResNet-style blocks with Fixup init |
| `SSGAN/models/Based_on_BRITS.py` | 3-player GAN (G, D, C) architecture |
| `TimesNet/models/TimesNet.py` | 2D convolution on periodic time series |

### Data Processing

| File | Purpose |
|------|---------|
| `R3GAN/dataset_tool.py` | Convert datasets to R3GAN format |
| `R3GAN/process_air_quality.py` | AirQuality-specific preprocessing |
| `SSGAN/preprocess.py` | SSGAN data preparation |
| `SSGAN/data_loader.py` | Load data with forward/backward passes |
| `TimesNet/data_provider/data_loader.py` | TimesNet data loading |

### GUI Components

| File | Description |
|------|-------------|
| `PURE-GUIv2/app_dad.py` | Main Dash app with routing |
| `pages/data_analysis_dad.py` | Data upload, visualization, statistics |
| `pages/time_imputation_dad.py` | Model selection and imputation |
| `pages/model_visualization.py` | Result comparison and plotting |

## Data Flow

### Training Pipeline

```
Raw Data (CSV/TXT)
    ↓
Preprocessing (dataset_tool.py / preprocess.py)
    ↓
Dataset Class (training/dataset.py / data_loader.py)
    ↓
DataLoader (batching, shuffling)
    ↓
Model Training (train.py / main.py / run.py)
    ↓
Checkpoints (*.pth / *.pkl)
    ↓
Evaluation (calc_metrics.py / metrics)
```

### Inference Pipeline

```
Incomplete Time Series
    ↓
Load Model Checkpoint
    ↓
Prepare Input (reshape, normalize, create mask)
    ↓
Generator Forward Pass
    ↓
Post-process (denormalize, reshape)
    ↓
Combine with Observed Values
    ↓
Imputed Time Series
```

### GUI Workflow

```
User Upload (CSV) → data_analysis_dad.py
    ↓
Visualize & Analyze
    ↓
Select Model → time_imputation_dad.py
    ↓
Load Checkpoint → model_files/
    ↓
Run Imputation
    ↓
Store Results → model_results/
    ↓
Compare Models → model_visualization.py
    ↓
Download Results
```

## Configuration Files

### R3GAN Presets
Defined in `train.py`:
- `AirQuality_MTSI`: Optimized for 36-feature air quality data
- `PSM_MTSI`: Optimized for 25-feature server metrics
- `PhysioNet_MTSI`: Optimized for 41-feature clinical data

### SSGAN Configuration
Set in `main.py`:
```python
choose = 0  # Dataset index
missing_rate = 50  # Missing percentage
dataset = 'AirQuality'
dimension = 36  # Number of features
```

### TimesNet Arguments
Via command-line arguments (see `run.py`):
- `--data`: Dataset name
- `--seq_len`: Sequence length
- `--enc_in/dec_in/c_out`: Feature dimensions
- `--mask_rate`: Missing data ratio

## Model Checkpoints

### Checkpoint Format

**R3GAN** (`.pkl`):
```python
{
    'G': Generator state dict,
    'D': Discriminator state dict,
    'G_ema': EMA generator state dict,
    'training_set_kwargs': {...},
    'augment_pipe': {...}
}
```

**SSGAN / TimesNet** (`.pth`):
```python
torch.save(model.state_dict(), 'model.pth')
```

### Loading Checkpoints

```python
# R3GAN
from R3GAN import legacy
with open('network-snapshot.pkl', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema']

# SSGAN / TimesNet
import torch
model = torch.load('model.pth')
model.load_state_dict(checkpoint)
```

## Output Directories

### Training Outputs
```
training_runs/
├── 00000-AirQuality/
│   ├── training_options.json        # Configuration
│   ├── log.txt                      # Training logs
│   ├── network-snapshot-*.pkl       # Checkpoints
│   ├── fakes*.png                   # Generated samples
│   └── metrics*.json                # Evaluation metrics
```

### GUI Outputs
```
uploaded_files/                      # User uploads
model_results/                       # Imputation results
    ├── R3GAN/
    ├── SSGAN/
    └── TimesNet/
```

## Important Notes

### Version Control
- `__pycache__/` directories are gitignored
- Large model files (`.pth`, `.pkl`) should not be committed
- Use Git LFS for datasets if needed

### Dependencies
- PyTorch 2.0+ for optimal performance
- CUDA 11.0+ for GPU acceleration
- See `requirements.txt` for complete list

### Development
- Use `PURE-GUIv2` for latest GUI features
- `PURE-GUI` kept for reference only
- Modify `TSImputation-master` for core algorithm changes

---

For detailed usage of each component, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

