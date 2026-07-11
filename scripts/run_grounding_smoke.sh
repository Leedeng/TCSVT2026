#!/bin/bash
#SBATCH --job-name=grnd_smoke
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/grnd_smoke_%j.out
#SBATCH --error=logs/grnd_smoke_%j.err

source ~/.bashrc
conda activate SPL2023
cd /scratch/project_2014500/dengli/TCSVT2026
echo '=== SMOKE: GRPO grounding permuted, 1 epoch ==='
python grpo_train_grounding.py --dataset iMiGUE \
    --reward_model_path /scratch/project_2018653/dengli/TCSVT2026/ckpt/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt \
    --sft_model_path sft_iMiGUE_Qwen2.5-0.5B \
    --epochs 1 --lr 1e-6 --G 8 --temperature 0.8 \
    --centroid_mode permuted --save_suffix grnd_smoke --ckpt_dir ckpt/iMiGUE/grounding_smoke
echo '=== SMOKE DONE ==='
