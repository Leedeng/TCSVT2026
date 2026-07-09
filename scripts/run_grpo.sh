#!/bin/bash
#SBATCH --job-name=TCSVT_grpo
#SBATCH --account=project_2010362
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:a100:1

source ~/.bashrc
conda activate SPL2023

DATASET=${1:-iMiGUE}
REWARD_MODEL=${2:-"best_model.pt"}
SFT_MODEL=${3:-"sft_iMiGUE_Qwen2.5-0.5B"}

echo "=========================================="
echo "GRPO Training - ${DATASET}"
echo "Reward model: ${REWARD_MODEL}"
echo "SFT model: ${SFT_MODEL}"
echo "=========================================="
python grpo_train.py \
    --dataset "$DATASET" \
    --reward_model_path "$REWARD_MODEL" \
    --sft_model_path "$SFT_MODEL" \
    --epochs 20 \
    --lr 1e-6 \
    --G 8 \
    --temperature 0.8

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
