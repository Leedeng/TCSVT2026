#!/bin/bash
#SBATCH --job-name=TCSVT2026
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

DATASET=${1:-SMG}
TRAIN_CSV=${2:-clip.csv}

echo "=========================================="
echo "Stage 1: Contrastive Learning - ${DATASET}"
echo "=========================================="
srun python train.py --dataset "$DATASET" --train_csv "$TRAIN_CSV"

# Find the best checkpoint from stage 1
BEST_CKPT=$(ls -t *_${DATASET}.pt 2>/dev/null | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "ERROR: No checkpoint found for ${DATASET}, skipping stage 2"
    exit 1
fi

echo "=========================================="
echo "Stage 2: Classifier Finetuning - ${DATASET}"
echo "Using checkpoint: ${BEST_CKPT}"
echo "=========================================="
srun python train_finetuned.py --dataset "$DATASET" --checkpoint "$BEST_CKPT"

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
