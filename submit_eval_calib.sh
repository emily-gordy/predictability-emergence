#!/bin/bash

#SBATCH --job-name=evalnn
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/evalnn_calib_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/evalnn_calib_%a.out
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emily.gordon@auckland.ac.nz
#SBATCH --time=1:00:00
#SBATCH --partition=milan

source ~/miniconda3/bin/activate
conda activate ml-env

cd /nesi/nobackup/uoa04506/predictability-emergence/predictability_emergence/

python3 eval_allvals.py --seed=$SLURM_ARRAY_TASK_ID --outputavgtime=5 --ssps=370
