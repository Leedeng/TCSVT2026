#!/bin/bash
#SBATCH --job-name=rsens
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/rsens_%x_%j.out
#SBATCH --error=logs/rsens_%x_%j.err

# Reward-sensitivity sweep on iMiGUE: GRPO refine -> generate descriptions -> retrain MCL.
# Usage:
#   sbatch --job-name=<cfg> scripts/run_reward_sens.sh <cfg> <alpha> <beta> <gamma> <minlen> <maxlen> <div> [grpo_ep] [retrain_ep]

source ~/.bashrc
conda activate SPL2023
cd /scratch/project_2014500/dengli/TCSVT2026

CFG=${1:-default}
ALPHA=${2:-1.0}
BETA=${3:-0.3}
GAMMA=${4:-0.2}
MINLEN=${5:-10}
MAXLEN=${6:-80}
DIV=${7:-0.5}
GRPO_EP=${8:-20}
RETRAIN_EP=${9:-40}

REWARD_MODEL=/scratch/project_2018653/dengli/TCSVT2026/ckpt/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt
SFT=sft_iMiGUE_Qwen2.5-0.5B
SENS_DIR=ckpt/iMiGUE/reward_sens
mkdir -p $SENS_DIR descriptions logs

echo "=== cfg=$CFG a=$ALPHA b=$BETA g=$GAMMA len=[$MINLEN,$MAXLEN] div=$DIV ==="

python grpo_train.py --dataset iMiGUE \
    --reward_model_path $REWARD_MODEL --sft_model_path $SFT \
    --epochs $GRPO_EP --lr 1e-6 --G 8 --temperature 0.8 \
    --alpha $ALPHA --beta_r $BETA --gamma $GAMMA \
    --r3_min_len $MINLEN --r3_max_len $MAXLEN --r3_div $DIV \
    --save_suffix sens_$CFG --ckpt_dir $SENS_DIR

python generate_grpo_descriptions.py --dataset iMiGUE \
    --sft_model_path $SENS_DIR/grpo_iMiGUE_sens_$CFG \
    --output descriptions/iMiGUE_sens_$CFG.json

python train_v2.py --dataset iMiGUE --use_descriptions \
    --desc_file descriptions/iMiGUE_sens_$CFG.json --epochs $RETRAIN_EP \
    --save_suffix iMiGUE_sens_$CFG --ckpt_dir $SENS_DIR \
    --log_dir ./log/iMiGUE_sens_$CFG

echo "=== DONE cfg=$CFG ==="
