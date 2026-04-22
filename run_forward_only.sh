#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/forward_only_%j.out
#SBATCH --error=logs/forward_only_%j.err
#SBATCH --job-name=fwdonly_vlm

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

python forward_only_vlm.py \
    --dataset iMiGUE \
    --model_path /scratch/project_2014500/dengli/EALD/QWEN_VL/Qwen2.5-VL-7B-Instruct \
    --num_frames 8 \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size 128 \
    --time_n 200 \
    --cache_dir feats_qwen_vl
