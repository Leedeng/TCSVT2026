#!/bin/bash
#SBATCH --job-name=ds_pipe
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/ds_pipe_%x_%j.out
#SBATCH --error=logs/ds_pipe_%x_%j.err

# R2-5: full pipeline on DeepSeek open-model seed corpus (SFT -> GRPO -> retrain).
# The frozen reward model is the same GPT-seed-trained MCL as the main run (judge
# held fixed; only the seed source changes). Usage:
#   sbatch scripts/run_deepseek_pipeline.sh [iMiGUE|SMG|MA52]

source ~/.bashrc
conda activate SPL2023
cd /scratch/project_2014500/dengli/TCSVT2026

DS=${1:-iMiGUE}
SEED=descriptions/${DS}_descriptions_deepseek.json
CKROOT=/scratch/project_2018653/dengli/TCSVT2026/ckpt
case $DS in
  iMiGUE) REWARD=$CKROOT/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt ;;
  SMG)    REWARD=$CKROOT/SMG/desc_v2/0.64_SMG_desc_v2.pt ;;
  MA52)   REWARD=$CKROOT/MA52/desc_v2/0.6_MA52_desc_v2.pt ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
OUT=ckpt/${DS}/deepseek
mkdir -p $OUT descriptions logs
echo "dataset=$DS  reward=$REWARD"

echo '===== 1) SFT Qwen on DeepSeek seeds ====='
python sft_train.py --dataset $DS --desc_file $SEED --save_suffix deepseek \
    --epochs 3 --batch_size 16 --lr 2e-5

echo '===== 2) GRPO refine (frozen reward model + DeepSeek SFT) ====='
python grpo_train.py --dataset $DS \
    --reward_model_path $REWARD \
    --sft_model_path sft_${DS}_deepseek \
    --epochs 20 --lr 1e-6 --G 8 --temperature 0.8 \
    --save_suffix deepseek --ckpt_dir $OUT

echo '===== 3) Generate descriptions from refined LoRA ====='
python generate_grpo_descriptions.py --dataset $DS \
    --sft_model_path $OUT/grpo_${DS}_deepseek \
    --output descriptions/${DS}_grpo_deepseek.json

echo '===== 4) Retrain MCL (80 ep) with refined DeepSeek descriptions ====='
python train_v2.py --dataset $DS --use_descriptions \
    --desc_file descriptions/${DS}_grpo_deepseek.json --epochs 80 \
    --save_suffix ${DS}_deepseek_final --ckpt_dir $OUT \
    --log_dir ./log/${DS}_deepseek_final

echo '===== DONE DeepSeek pipeline for '$DS' ====='
