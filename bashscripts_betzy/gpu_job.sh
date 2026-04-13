#!/bin/bash

#SBATCH --job-name=betzy_gpu_train
#SBATCH --account=nn9352k
#SBATCH --partition=accel
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --output=output/%j_gpu.txt
#SBATCH --error=error/%j_gpu.txt

set -euo pipefail

THREADS_PER_PROC=1

export OMP_NUM_THREADS="$THREADS_PER_PROC"
export MKL_NUM_THREADS="$THREADS_PER_PROC"
export OPENBLAS_NUM_THREADS="$THREADS_PER_PROC"
export NUMEXPR_NUM_THREADS="$THREADS_PER_PROC"
export PYTHONUNBUFFERED=1
export TRAIN_DEVICE=cuda

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error gpu_usage

CONFIG_PATH="${1:-template_configs/phnn_smoke.yml}"
CONTAINER="${CONTAINER:-$HOME/containers/pytorch_24.05-py3.sif}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing config file: $CONFIG_PATH" >&2
  exit 1
fi

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="singularity"
else
  echo "Missing apptainer/singularity in PATH. Load a container runtime module before submitting." >&2
  exit 1
fi

if [ ! -f "$CONTAINER" ]; then
  echo "Missing container at $CONTAINER. Set CONTAINER to an x86_64 CUDA/PyTorch SIF for Betzy." >&2
  exit 1
fi

"$CONTAINER_RUNTIME" exec --nv "$CONTAINER" python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('gpu_count', torch.cuda.device_count()); print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LOG="${GPU_LOG:-$SLURM_SUBMIT_DIR/gpu_usage/gpu_usage_${SLURM_JOB_ID}.csv}"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
    --format=csv -l 10 > "$GPU_LOG" &
  GPU_MON_PID=$!
  trap "kill $GPU_MON_PID" EXIT
fi

srun "$CONTAINER_RUNTIME" exec --nv "$CONTAINER" python train.py --config "$CONFIG_PATH"
