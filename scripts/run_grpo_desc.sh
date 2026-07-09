#!/bin/bash
#SBATCH --job-name=TCSVT_grpo_desc
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

echo "=========================================="
echo "E2E + GRPO Descriptions Training - ${DATASET}"
echo "=========================================="
python train.py --dataset "$DATASET" --use_descriptions \
    --desc_file "descriptions/${DATASET}_grpo_descriptions.json" \
    --save_suffix "${DATASET}_grpo_desc" \
    --log_dir "./log/${DATASET}_e2e_grpo_desc"

echo "=========================================="
echo "Done! ${DATASET}"
echo "=========================================="
