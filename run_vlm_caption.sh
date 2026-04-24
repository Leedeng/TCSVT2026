#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/vlm_caption_%j.out
#SBATCH --error=logs/vlm_caption_%j.err
#SBATCH --job-name=vlmcap

mkdir -p logs captions
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

QWEN=/scratch/project_2018653/dengli/EALD/QWEN_VL/Qwen2.5-VL-7B-Instruct
# Note: project path is _2018653 here -- iterate datasets:
QWEN=/scratch/project_2014500/dengli/EALD/QWEN_VL/Qwen2.5-VL-7B-Instruct
GEMMA=/scratch/project_2014500/dengli/gemma-4-E4B-it

for DS in iMiGUE SMG MA52; do
    for MODEL_TAG in qwen gemma; do
        if [ "$MODEL_TAG" = "qwen" ]; then
            MODEL_PATH=$QWEN
        else
            MODEL_PATH=$GEMMA
        fi
        OUT=captions/${DS}_${MODEL_TAG}.json
        echo "=== $DS / $MODEL_TAG -> $OUT ==="
        python vlm_caption.py \
            --dataset "$DS" \
            --model "$MODEL_TAG" \
            --model_path "$MODEL_PATH" \
            --k_per_class 3 \
            --num_frames 8 \
            --out "$OUT"
    done
done
