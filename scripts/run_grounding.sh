#!/bin/bash
#SBATCH --job-name=grnd
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/grnd_%x_%j.out
#SBATCH --error=logs/grnd_%x_%j.err

# Grounding control (R1-2/R2-4): GRPO with corrupted centroids -> generate -> retrain.
# Reuses the existing frozen reward model and GPT-seed SFT; only the centroid mode changes.
# Usage: sbatch --job-name=grnd_<mode> scripts/run_grounding.sh <permuted|random>

source ~/.bashrc
conda activate SPL2023
cd /scratch/project_2014500/dengli/TCSVT2026

MODE=${1:-permuted}
DS=iMiGUE
REWARD=/scratch/project_2018653/dengli/TCSVT2026/ckpt/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt
SFT=sft_iMiGUE_Qwen2.5-0.5B
OUT=ckpt/iMiGUE/grounding
mkdir -p $OUT descriptions logs

echo "===== GRPO grounding (mode=$MODE) ====="
python grpo_train_grounding.py --dataset $DS \
    --reward_model_path $REWARD --sft_model_path $SFT \
    --epochs 20 --lr 1e-6 --G 8 --temperature 0.8 \
    --centroid_mode $MODE \
    --save_suffix grnd_$MODE --ckpt_dir $OUT

echo "===== generate descriptions ====="
python generate_grpo_descriptions.py --dataset $DS \
    --sft_model_path $OUT/grpo_iMiGUE_grnd_$MODE \
    --output descriptions/iMiGUE_grnd_$MODE.json

echo "===== retrain MCL 80ep ====="
python train_v2.py --dataset $DS --use_descriptions \
    --desc_file descriptions/iMiGUE_grnd_$MODE.json --epochs 80 \
    --save_suffix iMiGUE_grnd_$MODE --ckpt_dir $OUT \
    --log_dir ./log/iMiGUE_grnd_$MODE

echo "===== DONE grounding $MODE ====="
