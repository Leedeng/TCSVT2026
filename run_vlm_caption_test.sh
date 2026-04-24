#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/vlmcap_test_%j.out
#SBATCH --error=logs/vlmcap_test_%j.err
#SBATCH --job-name=vlmcap_t

mkdir -p logs captions
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

QWEN=/scratch/project_2014500/dengli/EALD/QWEN_VL/Qwen2.5-VL-7B-Instruct
GEMMA=/scratch/project_2014500/dengli/gemma-4-E4B-it

# Smoke test: iMiGUE, k=1 per class, both Qwen and Gemma
for MODEL_TAG in qwen gemma; do
    if [ "$MODEL_TAG" = "qwen" ]; then
        MODEL_PATH=$QWEN
    else
        MODEL_PATH=$GEMMA
    fi
    OUT=captions/iMiGUE_${MODEL_TAG}_test.json
    echo "=== iMiGUE / $MODEL_TAG -> $OUT ==="
    python vlm_caption.py \
        --dataset iMiGUE \
        --model "$MODEL_TAG" \
        --model_path "$MODEL_PATH" \
        --k_per_class 1 \
        --num_frames 8 \
        --out "$OUT"
done
