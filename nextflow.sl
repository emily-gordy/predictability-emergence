#!/bin/bash -e

#SBATCH --account               nesi99999
#SBATCH --job-name              pred-em-nextflow
#SBATCH --output                "%x-%j.out"
#SBATCH --time                  08:00:00     # required walltime
#SBATCH --ntasks                1          # number of MPI tasks
#SBATCH --cpus-per-task         8   # number of threads per MPI task
#SBATCH --mem                   2GB

module purge && module load Miniforge3 && module load Nextflow/25.10.2

source $(conda info --base)/etc/profile.d/conda.sh && export PYTHONNOUSERSITE=1

conda activate /nesi/nobackup/nesi99999/jreeve/predictability-emergence/ml-env

export NFX_OPTS="-Xms=512m -Xmx=8g"

nextflow run predict-emergence.nf -profile test,mahuika