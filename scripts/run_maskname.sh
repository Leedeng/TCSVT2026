#!/bin/bash
#SBATCH --job-name=maskname
#SBATCH --account=project_2014500
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/maskname_%x_%j.out
#SBATCH --error=logs/maskname_%x_%j.err

# Class-name masking control (R1-2): mask the label string in the GRPO descriptions,
# retrain MCL, compare to unmasked (65.93). Usage: sbatch --job-name=mask_<mode> scripts/run_maskname.sh <phrase|word>
source ~/.bashrc
conda activate SPL2023
cd /scratch/project_2014500/dengli/TCSVT2026

MODE=${1:-phrase}
DS=iMiGUE
GRPO=descriptions/${DS}_grpo_descriptions.json
MASKED=descriptions/${DS}_grpo_mask${MODE}.json
OUT=ckpt/${DS}/maskname
mkdir -p $OUT descriptions logs

echo "===== mask class names (mode=$MODE) ====="
python mask_class_names.py --desc_file $GRPO --mode $MODE --output $MASKED

echo "===== retrain MCL 80ep on masked descriptions ====="
python train_v2.py --dataset $DS --use_descriptions \
    --desc_file $MASKED --epochs 80 \
    --save_suffix ${DS}_mask${MODE} --ckpt_dir $OUT \
    --log_dir ./log/${DS}_mask${MODE}

echo "===== DONE maskname $MODE ====="
