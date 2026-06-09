#!/bin/bash
#SBATCH --job-name=gpu_env_check
#SBATCH --account=nn9352k
#SBATCH --partition=accel
#SBATCH --qos=devel
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --output=output/%j_gpu_env_check.out
#SBATCH --error=error/%j_gpu_env_check.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error

module load NRIS/GPU
if ! command -v apptainer >/dev/null 2>&1; then
  module load apptainer || true
fi

CONTAINER="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
if [ ! -f "$CONTAINER" ]; then
  echo "Missing container at $CONTAINER. Set CONTAINER to your .sif path." >&2
  exit 1
fi

srun apptainer exec --nv "$CONTAINER" python - <<'PY'
import torch, numpy, scipy, matplotlib, sklearn, yaml, tensorboard
print("imports OK")
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
PY
