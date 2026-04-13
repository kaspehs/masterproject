#!/bin/bash
#SBATCH --job-name=betzy_gpu_env_check
#SBATCH --account=nn9352k
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=1
#SBATCH --output=output/%j_gpu_env_check.out
#SBATCH --error=error/%j_gpu_env_check.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error

ENV_PREFIX="${ENV_PREFIX:-$HOME/ml-env}"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "Missing virtualenv at $ENV_PREFIX. Set ENV_PREFIX to your Betzy env before submitting." >&2
  exit 1
fi

source "$ENV_PREFIX/bin/activate"

srun python - <<'PY'
import torch, numpy, scipy, matplotlib, sklearn, yaml, tensorboard
print("imports OK")
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("gpu0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
