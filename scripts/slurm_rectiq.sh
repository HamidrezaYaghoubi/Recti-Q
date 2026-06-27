#!/bin/bash
# Recti-Q experiment on the UMIACS Nexus Gamma cluster.
#
# Usage:
#   sbatch scripts/slurm_rectiq.sh configs/imagenet_c_rectiq.yaml
#   sbatch scripts/slurm_rectiq.sh configs/pacs_rectiq.yaml
#   CONDA_ENV=saficiency CONFIG_PATH=configs/pacs_rectiq.yaml sbatch scripts/slurm_rectiq.sh

#SBATCH --job-name=rectiq
#SBATCH --output=slurm_logs/%x.out.%j
#SBATCH --error=slurm_logs/%x.out.%j
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --time=1-00:00:00

set -euo pipefail

if [[ -n ${SLURM_SUBMIT_DIR:-} ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/slurm_logs" "$ROOT_DIR/logs"

# ── HuggingFace / cache on the project FS (never home or scratch) ──
export HF_HOME=/fs/nexus-projects/pc_driving/yaghoubi/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=/fs/nexus-projects/pc_driving/yaghoubi/torch_cache
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"

# ── Modules + conda env ──
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/12.6.3 || true
  module load gcc/11.2.0 || true
fi

CONDA_ENV="${CONDA_ENV:-saficiency}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
elif [[ -f /fs/nexus-scratch/yaghoubi/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /fs/nexus-scratch/yaghoubi/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
else
  echo "Conda not found; expected env: $CONDA_ENV" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-${CONFIG_PATH:-configs/imagenet_c_rectiq.yaml}}"
MODELS="${MODELS:-}"          # space-separated model names; empty = all in config
EXTRA_ARGS="${EXTRA_ARGS:-}"  # e.g. "--debug --no-wandb"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
TAG="$(basename "${CONFIG_PATH%.yaml}")${MODELS:+_${MODELS// /-}}"
RUN_LOG="$ROOT_DIR/logs/rectiq_${TAG}_${TIMESTAMP}.log"

RUN_CMD=("$PYTHON_BIN" -m src.main --config "$CONFIG_PATH")
[[ -n "$MODELS" ]] && RUN_CMD+=(--models $MODELS)
[[ -n "$EXTRA_ARGS" ]] && RUN_CMD+=($EXTRA_ARGS)

echo "Root dir:   $ROOT_DIR"
echo "Node:       $(hostname)"
echo "Conda env:  $CONDA_ENV"
echo "Config:     $CONFIG_PATH"
echo "Models:     ${MODELS:-<all>}"
echo "HF_HOME:    $HF_HOME"
echo "Run log:    $RUN_LOG"
echo "Command:    ${RUN_CMD[*]}"
nvidia-smi --list-gpus || true

"${RUN_CMD[@]}" 2>&1 | tee "$RUN_LOG"
