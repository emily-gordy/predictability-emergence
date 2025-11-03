#!/bin/bash

#SBATCH --job-name=hp_tune
#SBATCH --error=/nesi/nobackup/uoa04506/job_output/hp_tune_%a.err
#SBATCH --output=/nesi/nobackup/uoa04506/job_output/hp_tune_%a.out
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=A100:1
#SBATCH --array=0-9
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emily.gordon@auckland.ac.nz
#SBATCH --time=06:00:00
#SBATCH --partition=milan

module load Python/3.10.5-gimkl-2022a
module load CUDA/12.1.1
module load PyTorch/1.12.1-gimkl-2022a-Python-3.10.5-CUDA-11.6.2

cd /nesi/nobackup/uoa04506/predictability-emergence/

source /nesi/project/uoa04506/optuna_env/bin/activate

python3 -u hyperparamtertuning.py
