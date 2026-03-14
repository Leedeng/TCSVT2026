#!/bin/bash
#SBATCH --job-name=TCSVT2026
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:a100:1

source ~/.bashrc
conda activate SPL2023

DATASET=${1:-iMiGUE}
TRAIN_CSV=${2:-clip.csv}
CLS_WEIGHT=${3:-1.0}

echo "=========================================="
echo "End-to-End Training - ${DATASET}"
echo "cls_weight: ${CLS_WEIGHT}"
echo "=========================================="
python train.py --dataset "$DATASET" --train_csv "$TRAIN_CSV" --cls_weight "$CLS_WEIGHT"

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
