#!/bin/bash

#SBATCH --job-name=test_run_1
#SBATCH --account=nn9352k
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G

#SBATCH --output=output/%j.txt
#SBATCH --error=error/%j.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
export TRAIN_DEVICE=cuda


# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error

# Olivia: load GPU stack and run inside an ARM64 PyTorch container
module load NRIS/GPU

if ! command -v apptainer >/dev/null 2>&1; then
  module load apptainer || true
fi

CONTAINER="${CONTAINER:-/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif}"
if [ ! -f "$CONTAINER" ]; then
  echo "Missing container at $CONTAINER. Set CONTAINER to your .sif path." >&2
  exit 1
fi

echo "TRAIN_DEVICE=${TRAIN_DEVICE}"
apptainer exec --nv "$CONTAINER" python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('gpu_count', torch.cuda.device_count()); print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

srun apptainer exec --nv "$CONTAINER" python train.py --config runconfigs/phnn_smoke.yml
