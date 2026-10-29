#!/usr/bin/env bash
#SBATCH --job-name=equipment-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
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

# Interactive and batch jobs share module setup, venv activation, and GPU checks.
exec bash "$PWD/hpc/run.sh" "$@"
