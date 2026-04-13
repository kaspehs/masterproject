#!/bin/bash
#SBATCH --job-name=build_betzy_env
#SBATCH --account=nn9352k
#SBATCH --partition=preproc
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output/%j_build_betzy_env.out
#SBATCH --error=error/%j_build_betzy_env.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p output error

REQ_FILE="${REQ_FILE:-$SLURM_SUBMIT_DIR/requirements_olivia.txt}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/betzy-env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export http_proxy="${http_proxy:-http://10.63.2.48:3128/}"
export https_proxy="${https_proxy:-http://10.63.2.48:3128/}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN in PATH. Set PYTHON_BIN to a usable interpreter before submitting." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_PREFIX"
source "$ENV_PREFIX/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$REQ_FILE"

echo "Done. Activate with:"
echo "  source \"$ENV_PREFIX/bin/activate\""
