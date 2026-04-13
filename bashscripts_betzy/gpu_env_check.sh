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

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="singularity"
else
  echo "Missing apptainer/singularity in PATH. Load a container runtime module before submitting." >&2
  exit 1
fi

CONTAINER="${CONTAINER:-$HOME/containers/pytorch_24.05-py3.sif}"
if [ ! -f "$CONTAINER" ]; then
  echo "Missing container at $CONTAINER. Set CONTAINER to an x86_64 CUDA/PyTorch SIF for Betzy." >&2
  exit 1
fi

srun "$CONTAINER_RUNTIME" exec --nv "$CONTAINER" python - <<'PY'
import torch, numpy, scipy, matplotlib, sklearn, yaml, tensorboard
print("imports OK")
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("gpu0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
