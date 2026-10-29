#!/usr/bin/env bash
#SBATCH --job-name=equipment-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/equipment-%j.out
#SBATCH --error=logs/equipment-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root using sbatch}"
if [[ ! -f pyproject.toml || ! -d src/equipment_fitter ]]; then
    echo "Submit this job from the equipment-fitter project root." >&2
    exit 1
fi

# Optional user-provided module/virtualenv setup, executed on the compute node.
if [[ -n "${EQUIPMENT_ENV_SCRIPT:-}" ]]; then
    source "$EQUIPMENT_ENV_SCRIPT"
fi
EQUIPMENT_PYTHON="${EQUIPMENT_PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

EQUIPMENT_MANIFEST="${1:-data/lpc-glove/dataset.json}"
EQUIPMENT_CONFIG="${2:-configs/train-smoke.json}"
EQUIPMENT_RUN="runs/iris-${SLURM_JOB_ID}"

if [[ ! -f "$EQUIPMENT_MANIFEST" || ! -f "$EQUIPMENT_CONFIG" ]]; then
    echo "Prepare/transfer the dataset and configuration before submitting." >&2
    exit 1
fi

# This job never downloads data, packages, or model weights.
srun --ntasks=1 "$EQUIPMENT_PYTHON" -m equipment_fitter pilot \
    --manifest "$EQUIPMENT_MANIFEST" \
    --config "$EQUIPMENT_CONFIG" \
    --output "$EQUIPMENT_RUN" \
    --device cuda
