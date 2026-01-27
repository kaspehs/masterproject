#!/bin/bash
#SBATCH --job-name=phase_error_map
#SBATCH --account=nn9352k
#SBATCH --time=00:10:00
#SBATCH --partition=preproc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output/%j.txt
#SBATCH --error=error/%j.txt

set -o errexit
set -o nounset

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Unbuffered python prints
export PYTHONUNBUFFERED=1

# Force CPU on preproc
export TRAIN_DEVICE=cpu
export PHASE_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""

# Node-local tmp (SLURM_TMPDIR may be unset on some partitions)
export SLURM_TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}}"
mkdir -p "$SLURM_TMPDIR"

# Persistent matplotlib cache (build once)
export MPLCONFIGDIR="$HOME/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

module load Python/3.10.8-GCCcore-12.2.0
export PS1=\$
source "$HOME/ml-env/bin/activate"

# Copy heavy I/O inputs to node-local storage
mkdir -p "$SLURM_TMPDIR/groundtruth_runs_100hz" "$SLURM_TMPDIR/models"
cp -a Data_Gen/groundtruth_runs_100hz/. "$SLURM_TMPDIR/groundtruth_runs_100hz/"
cp -a models/. "$SLURM_TMPDIR/models/"

# phase_error_map.py parameters
export PHASE_MODEL_PATH="$SLURM_TMPDIR/models/vpinn_test2_0126-131515.pt"
export PHASE_RUNS_DIR="$SLURM_TMPDIR/groundtruth_runs_100hz"
export PHASE_LOGGER_HZ=100
export PHASE_STEADY_STATE_WINDOW_S=None
export PHASE_EVAL_BATCH_SIZE=512
export PHASE_PRINT_PER_RUN=0

python -u plotting_etc/phase_error_map.py