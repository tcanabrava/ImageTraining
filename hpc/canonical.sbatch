#!/usr/bin/env bash
#SBATCH --job-name=equipment-canonical
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/canonical-%j.out
#SBATCH --error=logs/canonical-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
# shellcheck source=hpc/environment.sh
source hpc/environment.sh
equipment_setup_environment "$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
srun "$EQUIPMENT_PYTHON" -m equipment_fitter.canonical_compare \
    --manifest "${1:-data/lpc-footwear-expanded/development.json}" \
    --output "runs/canonical-${SLURM_JOB_ID}" --device cuda --steps "${2:-1600}"
