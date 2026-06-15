#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate base
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像; 墙外可注释/改官方
export PYTHONPATH="$PROJ_DIR:$PYTHONPATH"
cd "$PROJ_DIR"

echo "============================================"
echo "Phase 1 Round 2: No augmentation experiments"
echo "============================================"

# Exp A: Electricity WITHOUT augmentation
echo ""
echo "[EXP A] Electricity no augment"
python train_refiner.py \
    --dataset Electricity --coarse linear --seq_len 96 \
    --epochs 50 --batch_size 64 \
    --width 128 --n_stages 3 --n_blocks 2 --cardinality 32 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --stride_divisor 4 \
    --outdir results/phase1_elec_noaug

# Exp B: Electricity with mean fill coarse
echo ""
echo "[EXP B] Electricity mean fill"
python train_refiner.py \
    --dataset Electricity --coarse mean --seq_len 96 \
    --epochs 50 --batch_size 64 \
    --width 128 --n_stages 3 --n_blocks 2 --cardinality 32 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 20.0 --lambda_freq 2.0 \
    --stride_divisor 4 \
    --outdir results/phase1_elec_mean

# Exp C: Weather with high recon weight (suppress adversarial)
echo ""
echo "[EXP C] Weather high recon weight"
python train_refiner.py \
    --dataset Weather --coarse linear --seq_len 96 \
    --epochs 200 --batch_size 32 \
    --width 64 --n_stages 3 --n_blocks 2 --cardinality 16 \
    --lr 1e-4 --gamma 0.5 --lambda_recon 50.0 --lambda_freq 5.0 \
    --stride_divisor 2 \
    --outdir results/phase1_weather_highrec

echo ""
echo "============================================"
echo "Round 2 complete! Summary:"
echo "============================================"
for d in results/phase1_*/; do
    if [ -f "${d}results.json" ]; then
        echo "--- $(basename $d) ---"
        python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
ds = r['dataset']
cm = r['coarse_method']
c = r['coarse_metrics']['MAE']
rf = r['refined_metrics']['MAE']
imp = r['improvement_pct']
print('  %s (%s): %.6f -> %.6f  (%+.2f%%)' % (ds, cm, c, rf, imp))
" "${d}results.json"
    fi
done
