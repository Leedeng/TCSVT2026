#!/bin/bash
#SBATCH --job-name=TCSVT2026_S2
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
CHECKPOINT=${2}

if [ -z "$CHECKPOINT" ]; then
    # Auto-find best checkpoint for this dataset
    CHECKPOINT=$(ls -t *_${DATASET}.pt 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found for ${DATASET}"
    exit 1
fi

echo "=========================================="
echo "Stage 2: Classifier Finetuning - ${DATASET}"
echo "Using checkpoint: ${CHECKPOINT}"
echo "=========================================="
python train_finetuned.py --dataset "$DATASET" --checkpoint "$CHECKPOINT"

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
