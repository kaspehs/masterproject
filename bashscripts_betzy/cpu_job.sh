#!/bin/bash

#SBATCH --job-name=betzy_cpu_train
#SBATCH --account=nn9352k
#SBATCH --partition=preproc
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --output=output/%j_cpu.txt
#SBATCH --error=error/%j_cpu.txt

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export TRAIN_DEVICE=cpu

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error

CONFIG_PATH="${1:-template_configs/phnn_smoke.yml}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/betzy-env}"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Missing virtualenv at $ENV_PREFIX. Build it with bashscripts_betzy/build_betzy_env.sh first." >&2
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing config file: $CONFIG_PATH" >&2
  exit 1
fi

source "$ENV_PREFIX/bin/activate"

srun python train.py --config "$CONFIG_PATH"
