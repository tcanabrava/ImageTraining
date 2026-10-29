#!/usr/bin/env bash
#SBATCH --job-name=equipment-appearance
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/appearance-%j.out
#SBATCH --error=logs/appearance-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
# shellcheck source=hpc/environment.sh
source hpc/environment.sh
equipment_setup_environment "$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
srun "$EQUIPMENT_PYTHON" -m equipment_fitter.appearance \
    --checkpoint "${1:?Provide the trained checkpoint}" \
    --manifest "${2:?Provide its benchmark manifest}" \
    --output "runs/appearance-${SLURM_JOB_ID}" --device cuda
