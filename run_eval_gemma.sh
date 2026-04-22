#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/eval_gemma_%j.out
#SBATCH --error=logs/eval_gemma_%j.err
#SBATCH --job-name=gemma_zs

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

# Gemma 4 needs transformers>=5.5.0.dev0; if loading fails, uncomment:
# pip install --user --upgrade git+https://github.com/huggingface/transformers@main

python eval_vlm_gemma.py \
    --dataset iMiGUE \
    --model_path /scratch/project_2014500/dengli/gemma-4-E4B \
    --num_frames 8 \
    --max_new_tokens 50
