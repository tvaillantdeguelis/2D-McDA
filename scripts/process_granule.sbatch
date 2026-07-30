#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G


CONFIG_FILE="$1"


python scripts/process_granule.py "${CONFIG_FILE}"
