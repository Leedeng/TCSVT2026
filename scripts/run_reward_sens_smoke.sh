#!/bin/bash
#SBATCH --job-name=rsens_smoke
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/rsens_smoke_%j.out
#SBATCH --error=logs/rsens_smoke_%j.err

# Smoke test: exercise the NEW r3_* CLI path quickly.
# GRPO 1 epoch with non-default thresholds + generate 2 descriptions/class.

source ~/.bashrc
conda activate SPL2023
cd /scratch/project_2014500/dengli/TCSVT2026

REWARD_MODEL=/scratch/project_2018653/dengli/TCSVT2026/ckpt/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt
SFT=sft_iMiGUE_Qwen2.5-0.5B
SENS_DIR=ckpt/iMiGUE/reward_sens_smoke
mkdir -p $SENS_DIR descriptions logs

echo '=== SMOKE: GRPO 1 epoch, len=[5,60] div=0.7, a=0.5 b=0.5 g=0.4 ==='
python grpo_train.py --dataset iMiGUE \
    --reward_model_path $REWARD_MODEL --sft_model_path $SFT \
    --epochs 1 --lr 1e-6 --G 8 --temperature 0.8 \
    --alpha 0.5 --beta_r 0.5 --gamma 0.4 \
    --r3_min_len 5 --r3_max_len 60 --r3_div 0.7 \
    --save_suffix smoke --ckpt_dir $SENS_DIR

echo '=== SMOKE: generate 2 descriptions/class ==='
python generate_grpo_descriptions.py --dataset iMiGUE \
    --sft_model_path $SENS_DIR/grpo_iMiGUE_smoke \
    --num_per_class 2 --output descriptions/iMiGUE_smoke.json

echo '=== SMOKE DONE ==='
