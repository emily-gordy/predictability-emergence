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

# path to the singularity image file (optionally replace with your own)
SIF=/opt/nesi/containers/lambda-stack/lambda-stack-focal-latest.sif

apptainer exec --nv -B ${PWD} ${SIF} python3 -u trainnn.py $SLURM_ARRAY_TASK_ID



# module load Python/3.10.5-gimkl-2022a
# module load CUDA/12.1.1
# module load PyTorch/1.12.1-gimkl-2022a-Python-3.10.5-CUDA-11.6.2

# cd /nesi/nobackup/uoa04506/predictability-emergence/

# python3 -u trainnn.py $SLURM_ARRAY_TASK_ID
