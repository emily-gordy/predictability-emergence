#!/bin/bash
#SBATCH --array=262
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/fixtrainnn_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/fixtrainnn_%a.out
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --time=0:30:00
#SBATCH --partition=milan

# Map array index to your custom values
#mapfile -t VALUES < jobint.txt
#MY_VALUE=${VALUES[$SLURM_ARRAY_TASK_ID-1]}

#echo "Processing value: $MY_VALUE"

source ~/miniconda3/bin/activate
conda activate ml-env

#python3 -u trainnn.py $MY_VALUE
python3 -u trainnn.py $SLURM_ARRAY_TASK_ID
