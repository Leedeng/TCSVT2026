#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=small
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/desc_examples_%j.out
#SBATCH --error=logs/desc_examples_%j.err
#SBATCH --job-name=descex

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

for DS in iMiGUE SMG MA52; do
    echo "=== ${DS} ==="
    python make_desc_examples.py \
        --dataset "$DS" \
        --k_per_class 10 \
        --n_frames 9 \
        --videos_from training_clips \
        --out_dir experiment/desc_examples
done
