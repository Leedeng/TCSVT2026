#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/coupled_pipe_%j.out
#SBATCH --error=logs/coupled_pipe_%j.err
#SBATCH --job-name=cpl_pipe

mkdir -p logs descriptions
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

# 1) Generate descriptions JSON from the coupled GRPO checkpoint
python generate_grpo_descriptions.py \
    --dataset iMiGUE \
    --sft_model_path ckpt_coupled/grpo_iMiGUE_coupled \
    --output descriptions/iMiGUE_grpo_coupled.json

# 2) Re-train MCL using the coupled descriptions
python train.py --dataset iMiGUE --use_descriptions \
    --desc_file descriptions/iMiGUE_grpo_coupled.json \
    --save_suffix iMiGUE_grpo_coupled \
    --log_dir ./log/iMiGUE_e2e_grpo_coupled
