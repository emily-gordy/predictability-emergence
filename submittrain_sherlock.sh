#!/bin/bash

#SBATCH --job-name=train_all_%a
#SBATCH --error=/scratch/users/egordon4/job_output/train_all_%a.err
#SBATCH --output=/scratch/users/egordon4/job_output/train_all_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=egordon4@stanford.edu
#SBATCH --time=8:00:00
#SBATCH -p serc

module load python/3.12.1
module load math
module load cuda/12.6.1
module load py-numpy/1.26.3_py312
module load py-pytorch/2.4.1_py312
module load py-scipy/1.12.0_py312

cd /scratch/users/egordon4/predictability-emergence/bin/

python3 -u trainnn.py $SLURM_ARRAY_TASK_ID