#!/bin/bash
#SBATCH --job-name=build_olivia_env
#SBATCH --account=nn9352k
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=output/%j_build_env.out
#SBATCH --error=error/%j_build_env.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
cd "$WORKDIR"
mkdir -p output error
REQ_FILE="${REQ_FILE:-$SCRIPT_DIR/requirements_olivia.txt}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/olivia-env}"

export http_proxy="${http_proxy:-http://10.63.2.48:3128/}"
export https_proxy="${https_proxy:-http://10.63.2.48:3128/}"

module load NRIS/CPU
module load hpc-container-wrapper

pip-containerize new --prefix "$ENV_PREFIX" --slim "$REQ_FILE"

echo "Done. Add to PATH:"
echo "  export PATH=\"$ENV_PREFIX/bin:\$PATH\""
