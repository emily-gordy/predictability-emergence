#!/bin/bash
#SBATCH --array=4,6,10,17,18,19
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/fixtrainnn_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/fixtrainnn_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=2:00:00
#SBATCH --partition=milan

# Map array index to your custom values
mapfile -t VALUES < jobint.txt
MY_VALUE=${VALUES[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing value: $MY_VALUE"

source ~/miniconda3/bin/activate
conda activate ml-env

python3 -u trainnn.py $MY_VALUE

