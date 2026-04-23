#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/finetune_gemma_%j.out
#SBATCH --error=logs/finetune_gemma_%j.err
#SBATCH --job-name=gemma_ft

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

python finetune_vlm_gemma.py \
    --dataset iMiGUE \
    --model_path /scratch/project_2014500/dengli/gemma-4-E4B-it \
    --epochs 3 \
    --batch_size 1 \
    --lr 2e-5 \
    --lora_r 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj \
    --save_dir vlm_ft_gemma_iMiGUE
