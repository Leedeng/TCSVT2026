#!/bin/bash
#SBATCH --job-name=TCSVT_sft
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:a100:1

source ~/.bashrc
conda activate SPL2023

DATASET=${1:-iMiGUE}

echo "=========================================="
echo "SFT Training - ${DATASET}"
echo "=========================================="
python sft_train.py --dataset "$DATASET" --epochs 10 --batch_size 16 --lr 2e-5

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
