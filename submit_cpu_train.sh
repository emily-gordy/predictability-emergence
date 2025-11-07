#!/bin/bash

#SBATCH --job-name=trainnn
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/trainnn_cpu_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/trainnn_cpu_%a.out
#SBATCH --array=0-9
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emily.gordon@auckland.ac.nz
#SBATCH --time=00:30:00
#SBATCH --partition=milan

source ~/miniconda3/bin/activate
conda activate ml-env

export DATALOADER_WORKERS=$((SLURM_CPUS_PER_TASK - 2))

python3 -u trainnn_clean.py $SLURM_ARRAY_TASK_ID
