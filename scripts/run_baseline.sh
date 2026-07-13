#!/bin/bash
#SBATCH --job-name=base
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/base_%x_%j.out
#SBATCH --error=logs/base_%x_%j.err

# Short-label MCL baseline (no descriptions), train_v2 to match the main pipeline.
# Usage: sbatch --job-name=base_<DS> scripts/run_baseline.sh <iMiGUE|SMG|MA52>
source ~/.bashrc
conda activate SPL2023
export PYTHONUNBUFFERED=1
cd /scratch/project_2014500/dengli/TCSVT2026

DS=${1:-SMG}
mkdir -p ckpt/baseline logs
echo "===== short-label baseline (train_v2, no descriptions): $DS ====="
python -u train_v2.py --dataset $DS --epochs 80 \
    --save_suffix $DS --ckpt_dir ckpt/baseline \
    --log_dir ./log/${DS}_baseline
echo "===== DONE baseline $DS ====="
