#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --job-name=extract

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

# Extract visual embeddings for each dataset's T^3 S (GRPO-refined MCL) ckpt.
# The ckpt_dir convention is project_2018653/.../ckpt/<dataset>/grpo_v2/<best>.pt .
CKPT_BASE=/scratch/project_2018653/dengli/TCSVT2026/ckpt

python extract_embeds.py \
    --dataset iMiGUE \
    --ckpt "${CKPT_BASE}/iMiGUE/grpo_v2/0.66_iMiGUE_grpo_v2.pt" \
    --out experiment/tsne/iMiGUE_ours.npz

python extract_embeds.py \
    --dataset SMG \
    --ckpt "${CKPT_BASE}/SMG/grpo_v2/0.67_SMG_grpo_v2.pt" \
    --out experiment/tsne/SMG_ours.npz

python extract_embeds.py \
    --dataset MA52 \
    --ckpt "${CKPT_BASE}/MA52/grpo_v2/0.61_MA52_grpo_v2.pt" \
    --out experiment/tsne/MA52_ours.npz
