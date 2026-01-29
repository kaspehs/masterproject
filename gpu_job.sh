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

module load Python/3.10.8-GCCcore-12.2.0

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# activate the virtual environment
source $HOME/ml-env/bin/activate

echo "TRAIN_DEVICE=${TRAIN_DEVICE}"
python -c "import sys, torch; print('python', sys.version); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('gpu_count', torch.cuda.device_count()); print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

python train.py --config runconfigs/phnn_smoke.yml
