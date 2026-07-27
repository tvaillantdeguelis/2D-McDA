#!/bin/bash

#SBATCH --job-name=McDA
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err


CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" yaml_files.txt)


cd /home/vaillant/codes/projects/2D_CALIOP/2D_McDA


python scripts/run.py \
    --config ${CONFIG}