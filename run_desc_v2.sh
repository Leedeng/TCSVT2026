#!/bin/bash
#SBATCH --job-name=TCSVT_v2
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
EPOCHS=${2:-40}

echo "=========================================="
echo "E2E v2 (L2norm + label_smooth) + Desc - ${DATASET} - ${EPOCHS} epochs"
echo "=========================================="
python train_v2.py --dataset "$DATASET" --use_descriptions --epochs "$EPOCHS" \
    --save_suffix "${DATASET}_desc_v2" \
    --log_dir "./log/${DATASET}_e2e_desc_v2"

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
