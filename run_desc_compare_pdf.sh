#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/desccmp_%j.out
#SBATCH --error=logs/desccmp_%j.err
#SBATCH --job-name=desccmp

mkdir -p logs experiment/desc_compare
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

for DS in iMiGUE SMG MA52; do
    OUT=experiment/desc_compare/${DS}_compare.pdf
    echo "=== $DS -> $OUT ==="
    python make_desc_compare_pdf.py \
        --dataset "$DS" \
        --out_pdf "$OUT"
done
