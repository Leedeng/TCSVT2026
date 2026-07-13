#!/bin/bash
#SBATCH --job-name=rwacc
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/rwacc_%j.out
#SBATCH --error=logs/rwacc_%j.err

# Per-class reward-accuracy correlation (R1-2), iMiGUE.
# Uses the newest short-label baseline in ckpt/baseline/ to also report the
# reward vs per-class accuracy GAIN (final - baseline).
source ~/.bashrc
conda activate SPL2023
export PYTHONUNBUFFERED=1
cd /scratch/project_2014500/dengli/TCSVT2026

CK=/scratch/project_2018653/dengli/TCSVT2026/ckpt
BASELINE=$(ls -t ckpt/baseline/*_iMiGUE.pt 2>/dev/null | head -1)
echo "baseline model: $BASELINE"

python -u reward_accuracy_corr.py --dataset iMiGUE \
    --reward_model $CK/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt \
    --final_model  $CK/iMiGUE/grpo_desc/0.65_iMiGUE_grpo_desc.pt \
    --baseline_model $BASELINE \
    --desc_file    descriptions/iMiGUE_grpo_descriptions.json \
    --output       reward_acc_iMiGUE.csv
