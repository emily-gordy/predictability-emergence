#!/bin/bash -e

#SBATCH --partition		milan
#SBATCH --job-name              pred-em-nextflow
#SBATCH --output                /nesi/nobackup/uoa04506/job_output/%x-%j.out
#SBATCH --time                  01:00:00     # required walltime
#SBATCH --ntasks                1          # number of MPI tasks
#SBATCH --cpus-per-task         8   # number of threads per MPI task
#SBATCH --mem                   2GB

module purge && module load Miniforge3 && module load Nextflow/25.10.2

source /home/egor650/miniconda3/bin/activate

conda activate ml-env

export NFX_OPTS="-Xms=512m -Xmx=8g"

nextflow run predict-emergence.nf -profile test,mahuika
