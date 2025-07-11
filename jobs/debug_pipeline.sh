#!/bin/bash
#SBATCH --job-name=hep_foundation_debug_simple
#SBATCH --account=m2616
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/debug-simple-%j.out
#SBATCH --error=logs/debug-simple-%j.err

# Print job information
echo "=========================================="
echo "DEBUG JOB (Simple) - Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Ensure we're in SCRATCH space
cd $SCRATCH/hep_foundation || {
    echo "ERROR: Could not change to $SCRATCH/hep_foundation"
    exit 1
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required NERSC modules
echo "Loading NERSC modules..."
module load craype
module load tensorflow/2.12.0

# Set environment variables
export NUMEXPR_MAX_THREADS=128
export CUDA_VISIBLE_DEVICES=0

# Verify GPU access
echo "Checking GPU access..."
nvidia-smi

# Ensure project dependencies are installed
echo "Ensuring project dependencies are installed..."
pip install --user -e .

# Remove any conflicting tensorflow installations
echo "Cleaning up any pip-installed tensorflow..."
pip uninstall --user -y tensorflow tensorflow-gpu tensorflow-cpu 2>/dev/null || true

# Verify environment
echo "Python version: $(python --version)"
echo "Python location: $(which python)"

# Test TensorFlow functionality
echo "Testing TensorFlow functionality..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CUDA available: {tf.test.is_built_with_cuda()}')
"

# Run pipeline in dry-run mode for testing
echo "Running pipeline in dry-run mode..."
echo "Time started: $(date)"

python scripts/run_pipelines.py --dry-run --max-configs 2

echo "=========================================="
echo "Debug job completed: $(date)"
echo "=========================================="
