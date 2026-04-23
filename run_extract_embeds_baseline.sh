#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/extract_baseline_%j.out
#SBATCH --error=logs/extract_baseline_%j.err
#SBATCH --job-name=extractB

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

CKPT_BASE=/scratch/project_2018653/dengli/TCSVT2026/ckpt

python extract_embeds.py \
    --dataset iMiGUE \
    --ckpt "${CKPT_BASE}/iMiGUE/desc_v2/0.58_iMiGUE_desc_v2.pt" \
    --out experiment/tsne/iMiGUE_baseline.npz

python extract_embeds.py \
    --dataset SMG \
    --ckpt "${CKPT_BASE}/SMG/desc_v2/0.53_SMG_desc_v2.pt" \
    --out experiment/tsne/SMG_baseline.npz

python extract_embeds.py \
    --dataset MA52 \
    --ckpt "${CKPT_BASE}/MA52/desc_v2/0.53_MA52_desc_v2.pt" \
    --out experiment/tsne/MA52_baseline.npz
