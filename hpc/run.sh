#!/usr/bin/env bash
# Run from an existing salloc allocation, or from iris-smoke.sbatch.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash hpc/run.sh [MANIFEST [CONFIG [OUTPUT]]]
       bash hpc/run.sh --check

Run inside a GPU Slurm allocation. Loads the documented Iris modules, creates and
activates .venv-iris using their packages, checks GPU execution, and runs the pilot.
Defaults: data/lpc-glove/dataset.json, configs/train-smoke.json, runs/iris-JOBID.
--check performs environment setup and GPU validation without training.

Overrides: EQUIPMENT_MODULES, EQUIPMENT_RESIF_ARCH, EQUIPMENT_VENV.
Existing custom environments: EQUIPMENT_ENV_SCRIPT and/or EQUIPMENT_PYTHON.
No packages, data, or weights are downloaded by this launcher.
HELP
    exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Request a GPU allocation with salloc first, or submit hpc/iris-smoke.sbatch." >&2
    exit 1
fi

equipment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# --overlap permits this step alongside the shell opened by salloc. Resources are
# still confined to this allocation; it does not request another GPU or another job.
exec srun --overlap --ntasks=1 --gpus-per-task=1 \
    bash "$equipment_root/hpc/run-worker.sh" "$@"
