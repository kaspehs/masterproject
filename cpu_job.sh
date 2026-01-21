#!/bin/bash

#SBATCH --job-name=test_run_1
#SBATCH --account=nn9352k
#SBATCH --time=00:05:00
#SBATCH --partition=preproc
#SBATCH --ntasks=1 --cpus-per-task=32
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

module load Python/3.10.8-GCCcore-12.2.0

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# activate the virtual environment
source $HOME/ml-env/bin/activate

python HNN.py --config HNNrunconfigs/pirate_final.yml
