#!/bin/bash
# Run experiments on SLURM cluster
#
# Usage:
#   sbatch scripts/slurm_run.sh configs/baseline_classification.yaml
#   sbatch scripts/slurm_run.sh configs/baseline_imagenet_c.yaml

#SBATCH --job-name=qda
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load modules (adjust for your cluster)
# module load cuda/12.6
# module load anaconda

# Activate environment
source activate quant 2>/dev/null || conda activate quant

# Get config from argument
CONFIG="${1:-configs/baseline_classification.yaml}"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# Navigate to project directory
cd /fs/nexus-projects/pc_driving/yaghoubi/quantization-decision-analysis

# Create logs directory
mkdir -p logs

# Run experiment
python -m src.main --config "$CONFIG"

echo "Job completed!"
