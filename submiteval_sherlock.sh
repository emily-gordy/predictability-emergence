#!/bin/bash

#SBATCH --job-name=eval_lat_%a
#SBATCH --error=/scratch/users/egordon4/job_output/eval_lat_%a.err
#SBATCH --output=/scratch/users/egordon4/job_output/eval_lat_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-16
#SBATCH --mem=10GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emily.gordon@auckland.ac.nz
#SBATCH --time=8:00:00
#SBATCH -p serc

module purge

module load python/3.12.1
module load math
module load py-numpy/1.26.3_py312
module load py-pytorch/2.4.1_py312
module load py-scipy/1.12.0_py312

cd /scratch/users/egordon4/predictability-emergence/bin/

# Define array of latitudes
LATS=(-70 -60 -50 -40 -30 -20 -10 0 10 20 30 40 50 60 70 80)

# Get the latitude for this task
LAT=${LATS[$SLURM_ARRAY_TASK_ID]}

python3 -u evalnn.py --lat=${LAT}