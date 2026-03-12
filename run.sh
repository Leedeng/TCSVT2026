#!/bin/bash
#SBATCH --job-name=SPL2023
#SBATCH --account=project_2006419
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:a100:1

source ~/.bashrc
conda activate SPL2023

DATASET=${1:-SMG}
TRAIN_CSV=${2:-clip.csv}

python train.py --dataset "$DATASET" --train_csv "$TRAIN_CSV"
