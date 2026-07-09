#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/grpo_gputest_%j.out
#SBATCH --error=logs/grpo_gputest_%j.err
#SBATCH --job-name=grpo_curve

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

# Move any existing GRPO logs aside so this run's events land in a fresh
# log/iMiGUE_grpo/ directory (grpo_train.py hardcodes that path).
if [ -d log/iMiGUE_grpo ]; then
    mv log/iMiGUE_grpo "log/iMiGUE_grpo_old_$(date +%s)"
fi

python grpo_train.py \
    --dataset iMiGUE \
    --reward_model_path /scratch/project_2018653/dengli/TCSVT2026/ckpt/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt \
    --sft_model_path /scratch/project_2014500/dengli/TCSVT2026/sft_iMiGUE_Qwen2.5-0.5B \
    --epochs 2 \
    --lr 1e-6 \
    --G 8 \
    --temperature 0.8
