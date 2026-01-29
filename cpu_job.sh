#!/bin/bash

#SBATCH --job-name=test_run_1
#SBATCH --account=nn9352k
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --nodes=1

#SBATCH --output=output/%j.txt
#SBATCH --error=error/%j.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export TRAIN_DEVICE=cpu


# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error

# Olivia: load a stack, then use a containerized Python env (hpc-container-wrapper)
module load NRIS/CPU

ENV_PREFIX="${ENV_PREFIX:-$HOME/olivia-env}"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Missing containerized env at $ENV_PREFIX. Create it with hpc-container-wrapper (conda-containerize new --prefix ...)." >&2
  exit 1
fi
export PATH="$ENV_PREFIX/bin:$PATH"

srun python train.py --config HNNrunconfigs/pirate_final.yml
