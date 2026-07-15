#!/bin/bash
#SBATCH --job-name=rwacc_iMiGUE
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/rwacc_iMiGUE_%j.out
#SBATCH --error=logs/rwacc_iMiGUE_%j.err

# iMiGUE reward-acc: Clip_label.csv was re-sorted, but the judge/final models are
# OLD order. Use the old-order label file (git HEAD) + an old-order (March) weak
# baseline so class indices align. Strip-insensitive desc lookup handles trailing spaces.
source ~/.bashrc
conda activate SPL2023
export PYTHONUNBUFFERED=1
cd /scratch/project_2014500/dengli/TCSVT2026

CK=/scratch/project_2018653/dengli/TCSVT2026/ckpt
python -u reward_accuracy_corr.py --dataset iMiGUE \
    --label_file iMiGUE_oldlabels.csv \
    --reward_model $CK/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt \
    --final_model  $CK/iMiGUE/grpo_desc/0.65_iMiGUE_grpo_desc.pt \
    --baseline_model ckpt/baseline/0.58_iMiGUE.pt \
    --desc_file descriptions/iMiGUE_grpo_descriptions.json \
    --output reward_acc_iMiGUE.csv
