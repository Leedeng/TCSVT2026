#!/bin/bash
#SBATCH --job-name=rwacc
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/rwacc_%x_%j.out
#SBATCH --error=logs/rwacc_%x_%j.err

# Per-class reward-accuracy correlation for SMG/MA52. Usage:
#   sbatch --job-name=rwacc_<DS> scripts/run_reward_acc_ds.sh <DS> <baseline_ckpt>
source ~/.bashrc
conda activate SPL2023
export PYTHONUNBUFFERED=1
cd /scratch/project_2014500/dengli/TCSVT2026

DS=$1
BASELINE=$2
CK=/scratch/project_2018653/dengli/TCSVT2026/ckpt
case $DS in
  SMG)  JUDGE=$CK/SMG/desc_v2/0.6_SMG_desc_v2.pt;   FINAL=$CK/SMG/grpo_v2/0.67_SMG_grpo_v2.pt ;;
  MA52) JUDGE=$CK/MA52/desc_v2/0.6_MA52_desc_v2.pt; FINAL=$CK/MA52/grpo_v2/0.6_MA52_grpo_v2.pt ;;
esac
echo "DS=$DS judge=$JUDGE final=$FINAL baseline=$BASELINE"
python -u reward_accuracy_corr.py --dataset $DS \
    --reward_model $JUDGE --final_model $FINAL \
    --baseline_model $BASELINE \
    --desc_file descriptions/${DS}_grpo_descriptions.json \
    --output reward_acc_${DS}.csv
