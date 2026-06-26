#!/bin/bash

#SBATCH --job-name=q_bdd_rectiq_off
#SBATCH --output=slurm_logs/%x.out.%j
#SBATCH --error=slurm_logs/%x.out.%j
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --time=2-00:00:00

set -euo pipefail

if [[ -n ${SLURM_SUBMIT_DIR:-} ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/slurm_logs" "$ROOT_DIR/logs"

if command -v module >/dev/null 2>&1 || [[ -f /etc/profile.d/modules.sh ]]; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.6.3
    module load gcc/11.2.0
  fi
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate quant
elif [[ -f /fs/nexus-scratch/yaghoubi/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /fs/nexus-scratch/yaghoubi/anaconda3/etc/profile.d/conda.sh
  conda activate quant
else
  echo "Conda not found; expected env: quant" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/quantize_bdd100k_yolo_finetuned_rectiq_disabled.yaml}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_LOG="$ROOT_DIR/logs/quant_bdd_rectiq_disabled_${TIMESTAMP}.log"

echo "Root dir: $ROOT_DIR"
echo "Using config: $CONFIG_PATH"
echo "Logging run output to: $RUN_LOG"
echo "Visible GPUs:"
nvidia-smi --list-gpus || true

"$PYTHON_BIN" -m src.main --config "$CONFIG_PATH" 2>&1 | tee "$RUN_LOG"
