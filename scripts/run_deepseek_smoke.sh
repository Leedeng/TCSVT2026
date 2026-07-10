#!/bin/bash
#SBATCH --job-name=ds_smoke
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/ds_smoke_%j.out
#SBATCH --error=logs/ds_smoke_%j.err

# Smoke: validate the new sft_train.py --desc_file/--save_suffix plumbing on DeepSeek seeds.
source ~/.bashrc
conda activate SPL2023
cd /scratch/project_2014500/dengli/TCSVT2026

echo '=== SMOKE: SFT 1 epoch on DeepSeek seeds ==='
python sft_train.py --dataset iMiGUE \
    --desc_file descriptions/iMiGUE_descriptions_deepseek.json \
    --save_suffix deepseek_smoke --epochs 1 --batch_size 16 --lr 2e-5
echo '=== SMOKE: check saved LoRA ==='
ls -la sft_iMiGUE_deepseek_smoke/ 2>/dev/null | head
echo '=== SMOKE DONE ==='
