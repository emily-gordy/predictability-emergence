#!/bin/bash

#SBATCH --job-name=evalcalib
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/evalnn_calib_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/evalnn_calib_%a.out
#SBATCH --array=0-31
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emily.gordon@auckland.ac.nz
#SBATCH --time=1:00:00
#SBATCH --partition=genoa,milan

source ~/miniconda3/bin/activate
conda activate ml-env

cd /nesi/nobackup/uoa04506/predictability-emergence/predictability_emergence/

# tasks 0-15: best 3 seeds, tasks 16-31: all 10 seeds, one latitude (-70 to 80) per task
NLATS=16
if [ $SLURM_ARRAY_TASK_ID -lt $NLATS ]; then
    NBEST=3
else
    NBEST=10
fi
LAT=$(( -70 + (SLURM_ARRAY_TASK_ID % NLATS) * 10 ))

python3 eval_calib_ensemble.py --lat=$LAT --n_best=$NBEST --outputavgtime=5 --calib_members 38 44
