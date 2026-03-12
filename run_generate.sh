#!/bin/bash
#SBATCH --job-name=gen_desc
#SBATCH --account=project_2014500
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000

source ~/.bashrc
conda activate SPL2023

cd /scratch/project_2014500/dengli/TCSVT2026

python generate_descriptions.py --dataset SMG --num_per_angle 5
python generate_descriptions.py --dataset iMiGUE --num_per_angle 5
python generate_descriptions.py --dataset MA52 --num_per_angle 5
