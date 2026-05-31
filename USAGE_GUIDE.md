# MTSIR3-GAN Usage Guide

Complete guide for running and training all models in this project.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start with GUI](#quick-start-with-gui)
3. [Training MTSIR3-GAN (R3GAN)](#training-mtsir3-gan-r3gan)
4. [Training SSGAN](#training-ssgan)
5. [Training TimesNet](#training-timesnet)
6. [Model Inference](#model-inference)
7. [Evaluation](#evaluation)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/he-yufeng/MTSIR3-GAN.git
cd MTSIR3-GAN
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n mtsir3gan python=3.8
conda activate mtsir3gan
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```python
import torch
import dash
import numpy as np
import pandas as pd
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS (Apple) available:", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
```

---

## Quick Start with GUI

### Launch the Web Interface
```bash
cd PURE-GUIv2
python app_dad.py
```

Open browser and go to: `http://127.0.0.1:8050`

### GUI Features

#### 1. Data Analysis Module
- **Upload Data**: Drag & drop CSV files or browse
- **Visualize**: Interactive plots with Plotly
- **Missing Patterns**: Analyze missing data distribution
- **Statistics**: View summary statistics

#### 2. Data Imputation Module
- **Select Model**: Choose between R3GAN, SSGAN, or TimesNet
- **Configure**: Adjust parameters (optional)
- **Run**: Execute imputation with one click
- **Download**: Save imputed results

#### 3. Model Visualization Module
- **Compare**: Side-by-side model comparison
- **Metrics**: View MAE, MSE, RMSE
- **Plots**: Generate violin plots, line plots
- **Export**: Save figures and metrics

### Example Workflow
1. Click "Data Analysis" → Upload `pm25_missing.csv`
2. Review data structure and missing patterns
3. Click "Data Imputation" → Select "MTSIR3-GAN"
4. Click "Run Imputation" → Wait for results
5. Click "Model Visualization" → Compare with other models
6. Download imputed data and visualizations

---

## Training MTSIR3-GAN (R3GAN)

### Basic Training

Navigate to R3GAN directory:
```bash
cd R3GAN
```

#### Train on AirQuality Dataset
```bash
python train.py \
    --outdir=./training_runs \
    --data=../datasets/AirQuality/pm25_missing.txt \
    --gpus=1 \
    --batch=64 \
    --kimg=10000 \
    --gamma=0.5 \
    --preset=AirQuality_MTSI
```

#### Train on PSM Dataset
```bash
python train.py \
    --outdir=./training_runs \
    --data=../datasets/PSM/train.csv \
    --gpus=1 \
    --batch=32 \
    --kimg=20000 \
    --gamma=1.0 \
    --preset=PSM_MTSI
```

#### Train on PhysioNet Dataset
```bash
python train.py \
    --outdir=./training_runs \
    --data=../datasets/PhysioNet/processed_data.txt \
    --gpus=1 \
    --batch=16 \
    --kimg=30000 \
    --gamma=0.8 \
    --preset=PhysioNet_MTSI
```

### Advanced Configuration

#### Custom Architecture
```bash
python train.py \
    --outdir=./training_runs \
    --data=../datasets/AirQuality/pm25_missing.txt \
    --gpus=1 \
    --batch=64 \
    --gamma=0.5 \
    --cbase=512 \                    # Base channel count
    --cmax=1024 \                    # Max channels
    --map-depth=2 \                  # Mapping network depth
    --mbstd-group=4                  # Minibatch std group size
```

#### Multi-GPU Training
```bash
python train.py \
    --outdir=./training_runs \
    --data=../datasets/PSM/train.csv \
    --gpus=4 \
    --batch=128 \
    --kimg=50000
```

#### Resume Training
```bash
python train.py \
    --outdir=./training_runs \
    --data=../datasets/AirQuality/pm25_missing.txt \
    --resume=./training_runs/00000-AirQuality/network-snapshot-001000.pkl \
    --gpus=1 \
    --batch=64
```

### Hyperparameter Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--batch` | Batch size | 32 | 16-128 |
| `--gamma` | R1/R2 regularization | 0.5 | 0.05-2.0 |
| `--kimg` | Training duration (k images) | 10000 | 1000-100000 |
| `--lr` | Learning rate | 0.0002 | 1e-5 to 1e-3 |
| `--cbase` | Base channels | 32768 | 16384-65536 |
| `--preset` | Configuration preset | None | AirQuality_MTSI, PSM_MTSI |

### Monitoring Training

#### TensorBoard
```bash
tensorboard --logdir=./training_runs
```

#### Weights & Biases
```bash
wandb login
# Training will automatically log to W&B
```

### Generate Samples
```bash
python gen_timeseries.py \
    --network=./training_runs/00000-AirQuality/network-snapshot-010000.pkl \
    --seeds=0-999 \
    --outdir=./generated_samples
```

---

## Training SSGAN

Navigate to SSGAN directory:
```bash
cd SSGAN
```

### Prepare Data
```bash
# Edit preprocess.py to set your dataset path
python preprocess.py
```

### Configure Training

Edit `main.py` to set dataset parameters:
```python
choose = 0  # Dataset index
missing_rate = 50  # Missing percentage
dataset = 'AirQuality'  # Dataset name
dimension = 36  # Number of features
```

### Run Training
```bash
python main.py \
    --epochs=50 \
    --batch_size=64 \
    --model=Based_on_BRITS
```

### Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Number of epochs | 50 |
| `--batch_size` | Batch size | 64 |
| `--model` | Model architecture | Based_on_BRITS |
| `--hid_size` | Hidden size | 108 (auto) |
| `--impute_weight` | Imputation loss weight | 1.0 |
| `--label_weight` | Classification loss weight | 1.0 |

### Model Architectures
- `Based_on_BRITS`: Semi-supervised with BRITS backbone
- `brits`: Original BRITS model
- `gru_d`: GRU-D baseline

---

## Training TimesNet

Navigate to TimesNet directory:
```bash
cd TimesNet
```

### Basic Training

#### PSM Dataset (25% Missing)
```bash
python run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./datasets/PSM/ \
    --data_path train.csv \
    --model_id PSM_mask_0.25 \
    --model TimesNet \
    --data PSM \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 0 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 25 \
    --dec_in 25 \
    --c_out 25 \
    --d_model 64 \
    --d_ff 64 \
    --top_k 3 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --mask_rate 0.25
```

#### AirQuality Dataset
```bash
python run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./datasets/AirQuality/ \
    --data_path pm25_missing.txt \
    --model_id AirQuality_mask_0.3 \
    --model TimesNet \
    --data AirQuality \
    --features M \
    --seq_len 144 \
    --enc_in 36 \
    --dec_in 36 \
    --c_out 36 \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 20 \
    --mask_rate 0.3
```

### Key Parameters

| Parameter | Description | PSM | AirQuality |
|-----------|-------------|-----|------------|
| `--seq_len` | Input sequence length | 96 | 144 |
| `--enc_in` | Encoder input size | 25 | 36 |
| `--dec_in` | Decoder input size | 25 | 36 |
| `--c_out` | Output size | 25 | 36 |
| `--d_model` | Model dimension | 64 | 128 |
| `--batch_size` | Batch size | 16 | 32 |
| `--mask_rate` | Missing rate | 0.25 | 0.3 |
| `--train_epochs` | Training epochs | 10 | 20 |

### Evaluation Only
```bash
python run.py \
    --task_name imputation \
    --is_training 0 \
    --model_id PSM_mask_0.25 \
    --model TimesNet \
    --data PSM \
    --root_path ./datasets/PSM/ \
    --data_path train.csv \
    --mask_rate 0.25
```

---

## Model Inference

### Python API

#### R3GAN Inference
```python
import torch
import numpy as np
from R3GAN import legacy

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('training_runs/00000-AirQuality/network-snapshot-010000.pkl', 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

# Prepare data
incomplete_data = np.load('incomplete.npy')  # Shape: (N, C, H, W)
mask = np.load('mask.npy')  # 1=observed, 0=missing

# Generate
batch_size = incomplete_data.shape[0]
z = torch.randn(batch_size, G.z_dim, device=device)
condition = torch.from_numpy(incomplete_data * mask).to(device).float()

with torch.no_grad():
    imputed = G(z, condition).cpu().numpy()

# Combine
result = incomplete_data * mask + imputed * (1 - mask)
```

#### SSGAN Inference
```python
import torch
from models import Based_on_BRITS

# Load model
model = torch.load('saved_models/SSGAN_AirQuality.pth')
model.eval()

# Prepare data (as dict with forward/backward)
data = {
    'forward': {'values': incomplete_tensor, 'masks': mask_tensor, ...},
    'backward': {'values': incomplete_tensor_rev, 'masks': mask_tensor_rev, ...}
}

# Impute
with torch.no_grad():
    ret = model.run_on_batch(data)
    imputed = ret['imputations']
```

#### TimesNet Inference
```python
from exp.exp_imputation import Exp_Imputation
import argparse

# Configure
args = argparse.Namespace(
    task_name='imputation',
    model='TimesNet',
    data='PSM',
    root_path='./datasets/PSM/',
    data_path='test.csv',
    model_id='PSM_mask_0.25',
    # ... other args
)

# Load experiment
exp = Exp_Imputation(args)
exp.test(setting='PSM_test', test=1)
```

### Batch Processing
```bash
# Create batch inference script
cat > batch_impute.py << 'EOF'
import glob
import numpy as np
from inference_utils import load_model, impute

model = load_model('path/to/model.pth')
files = glob.glob('data/*.npy')

for f in files:
    data = np.load(f)
    result = impute(model, data)
    np.save(f'imputed_{f}', result)
    print(f'Processed: {f}')
EOF

python batch_impute.py
```

---

## Evaluation

### Compute Metrics
```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_imputation(true, imputed, mask):
    """
    Evaluate imputation quality on masked regions.
    
    Args:
        true: Ground truth values
        imputed: Imputed values
        mask: 0 where data was missing, 1 where observed
    """
    # Only evaluate on masked (originally missing) positions
    missing_mask = (mask == 0)
    
    true_missing = true[missing_mask]
    imputed_missing = imputed[missing_mask]
    
    mae = mean_absolute_error(true_missing, imputed_missing)
    mse = mean_squared_error(true_missing, imputed_missing)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }

# Example usage
metrics = evaluate_imputation(ground_truth, imputed_data, mask_matrix)
print(f"MAE: {metrics['MAE']:.4f}")
print(f"MSE: {metrics['MSE']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
```

### Compare Models
```python
import pandas as pd

results = {
    'Model': ['MTSIR3-GAN', 'SSGAN', 'TimesNet'],
    'MAE': [0.412, 0.435, 0.396],
    'MSE': [0.252, 0.288, 0.265],
    'RMSE': [0.502, 0.537, 0.515]
}

df = pd.DataFrame(results)
print(df)
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch=16  # instead of 64

# Or use gradient accumulation
python train.py --batch=16 --grad-accum=4
```

#### 2. Module Not Found
```bash
# Ensure you're in the right directory
cd R3GAN

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. Data Loading Errors
```python
# Check data format
import numpy as np
data = np.loadtxt('dataset.txt', delimiter=',')
print(data.shape)  # Should match expected dimensions
```

#### 4. GUI Not Starting
```bash
# Check port availability
lsof -i :8050

# Use different port
python app_dad.py --port=8051
```

#### 5. Slow Training
```bash
# Enable mixed precision
python train.py --fp16=1

# Use multiple workers
python train.py --num-workers=8
```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

For R3GAN:
```bash
python train.py --debug=1 --dry-run=1
```

### Performance Optimization

#### Use Compiled Models (PyTorch 2.0+)
```python
import torch
model = torch.compile(model)
```

#### Enable CuDNN Benchmarking
```python
torch.backends.cudnn.benchmark = True
```

#### Data Loading
```python
DataLoader(dataset, num_workers=8, pin_memory=True)
```

---

## Additional Resources

- **Documentation**: See `docs/` folder for detailed API reference
- **Examples**: Check `examples/` for Jupyter notebooks
- **Issues**: Report bugs on [GitHub Issues](https://github.com/he-yufeng/MTSIR3-GAN/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/he-yufeng/MTSIR3-GAN/discussions)

---

## Support

If you encounter any issues:

1. Check this guide first
2. Search existing [GitHub Issues](https://github.com/he-yufeng/MTSIR3-GAN/issues)
3. Ask in [GitHub Discussions](https://github.com/he-yufeng/MTSIR3-GAN/discussions)
4. Open a new issue with:
   - Python version
   - PyTorch version
   - CUDA version (if using GPU)
   - Full error message
   - Steps to reproduce

---

**Happy Imputing! 🚀**

