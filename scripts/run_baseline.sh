#!/bin/bash
# Run baseline FP32 inference on ImageNet validation set
#
# Usage:
#   ./scripts/run_baseline.sh
#   ./scripts/run_baseline.sh --debug  # Quick test with 100 samples
#   ./scripts/run_baseline.sh --config configs/baseline_imagenet_c.yaml

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Activate conda environment
source activate quant 2>/dev/null || conda activate quant 2>/dev/null || echo "Conda env activation skipped"

# Default configuration
CONFIG="${CONFIG:-configs/baseline_classification.yaml}"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --debug)
            EXTRA_ARGS="$EXTRA_ARGS --debug"
            shift
            ;;
        --no-wandb)
            EXTRA_ARGS="$EXTRA_ARGS --no-wandb"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "========================================"
echo "Quantization Decision Analysis"
echo "Baseline FP32 Inference"
echo "========================================"
echo "Config: $CONFIG"
echo "Extra args: $EXTRA_ARGS"
echo "========================================"

# Run the experiment
python -m src.main --config "$CONFIG" $EXTRA_ARGS

echo "========================================"
echo "Experiment completed!"
echo "========================================"
