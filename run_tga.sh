#!/bin/bash
#SBATCH --job-name=TCSVT_TGA
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:a100:1

source ~/.bashrc
conda activate SPL2023

DATASET=${1:-iMiGUE}
TRAIN_CSV=${2:-clip.csv}

echo "=========================================="
echo "E2E + TGA Training - ${DATASET}"
echo "=========================================="
python train.py --dataset "$DATASET" --train_csv "$TRAIN_CSV" --use_tga

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
