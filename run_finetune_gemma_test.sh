#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/ftgemma_test_%j.out
#SBATCH --error=logs/ftgemma_test_%j.err
#SBATCH --job-name=gemma_t

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

# Smoke test: 30 training steps, no eval, log every 10 steps with peak mem
python finetune_vlm_gemma.py \
    --dataset iMiGUE \
    --model_path /scratch/project_2014500/dengli/gemma-4-E4B-it \
    --epochs 1 \
    --batch_size 1 \
    --lr 2e-5 \
    --lora_r 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj \
    --max_train_steps 30 \
    --skip_eval \
    --save_dir vlm_ft_gemma_iMiGUE_test
