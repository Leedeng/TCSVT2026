#!/bin/bash
#SBATCH --account=project_2014500
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/video_only_test_%j.out
#SBATCH --error=logs/video_only_test_%j.err
#SBATCH --job-name=vidonly_t

mkdir -p logs
source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

# Smoke test: 3 epochs on iMiGUE so we can confirm the pipeline and
# get a rough sense of per-epoch time before committing gpusmall to
# the full 50-epoch * 3-dataset run.
python train_video_only.py \
    --dataset iMiGUE \
    --epochs 3 \
    --lr 1e-4 \
    --batch_size 32 \
    --ckpt_dir ckpt_video_only/iMiGUE_test \
    --log_dir log/iMiGUE_video_only_test
