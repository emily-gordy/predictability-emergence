#!/bin/bash

#SBATCH --job-name=trainnn
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/trainnn_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/trainnn_%a.out
#SBATCH --array=73-288
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=40GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emily.gordon@auckland.ac.nz
#SBATCH --time=4:00:00
#SBATCH --partition=milan

source ~/miniconda3/bin/activate
conda activate ml-env
cd /nesi/nobackup/uoa04506/predictability-emergence/predictability_emergence/
python3 nosst_trainnn.py --seed=$SLURM_ARRAY_TASK_ID --outputavgtime=10