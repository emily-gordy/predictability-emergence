#!/bin/bash

#SBATCH --job-name=trainnn
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/trainnn_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/trainnn_%a.out
#SBATCH --ntasks=1
#SBATCH --array=0-9
#SBATCH --gpus-per-task=A100:1
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emily.gordon@auckland.ac.nz
#SBATCH --time=00:30:00
#SBATCH --partition=milan

module purge

module load Python/3.10.5-gimkl-2022a
source /nesi/project/uoa04506/ml_venv/bin/activate

python3 -u trainnn.py $SLURM_ARRAY_TASK_ID

