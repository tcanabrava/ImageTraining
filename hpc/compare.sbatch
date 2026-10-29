#!/usr/bin/env bash
#SBATCH --job-name=equipment-compare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/compare-%j.out
#SBATCH --error=logs/compare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
# shellcheck source=hpc/environment.sh
source hpc/environment.sh
equipment_setup_environment "$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
srun "$EQUIPMENT_PYTHON" -m equipment_fitter.compare \
    --manifest "${1:-data/lpc-footwear/dataset.json}" \
    --output "runs/compare-${SLURM_JOB_ID}" --device cuda --steps "${2:-200}"
