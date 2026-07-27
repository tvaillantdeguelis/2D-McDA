#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --mem=10000

# Activate Python virtual environment
mamba activate 2D_McDA

# Run program
python run.py --config "$1"
