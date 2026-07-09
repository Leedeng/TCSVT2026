#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=small
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/motiv_pdf_%j.out
#SBATCH --error=logs/motiv_pdf_%j.err
#SBATCH --job-name=motiv_pdf

mkdir -p logs experiment
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

python make_motivation_pdf.py \
    --datasets iMiGUE SMG MA52 \
    --top_n 30 \
    --out_pdf experiment/motivation_candidates.pdf
