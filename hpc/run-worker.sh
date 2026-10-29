#!/usr/bin/env bash
set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "This worker requires a GPU Slurm allocation." >&2; exit 1; }
equipment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$equipment_root"
if [[ "$#" -gt 3 || ( "${1:-}" == --* && "${1:-}" != "--check" ) || ( "${1:-}" == "--check" && "$#" -ne 1 ) ]]; then
    echo "Usage: bash hpc/run.sh [MANIFEST [CONFIG [OUTPUT]]] or --check" >&2
    exit 1
fi

equipment_manifest="${1:-data/lpc-glove/dataset.json}"
equipment_config="${2:-configs/train-smoke.json}"
if [[ "${1:-}" != "--check" && ( ! -f "$equipment_manifest" || ! -f "$equipment_config" ) ]]; then
    echo "Dataset/config missing. Prepare or transfer $equipment_manifest and $equipment_config first." >&2
    exit 1
fi

# shellcheck source=hpc/environment.sh
source "$equipment_root/hpc/environment.sh"
equipment_setup_environment "$equipment_root"
export PYTHONPATH="$equipment_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"

"$EQUIPMENT_PYTHON" "$equipment_root/hpc/preflight.py"
if [[ "${1:-}" == "--check" ]]; then
    echo "Environment and GPU check passed. Run 'bash hpc/run.sh' to train."
    exit 0
fi

equipment_run="${3:-runs/iris-${SLURM_JOB_ID}}"
if [[ "$#" -lt 3 && -e "$equipment_run" ]]; then
    equipment_run="${equipment_run}-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}"
fi
echo "Python: $EQUIPMENT_PYTHON"
echo "Saving this run to: $equipment_run (working directory: $equipment_root)"
exec "$EQUIPMENT_PYTHON" -m equipment_fitter pilot \
    --manifest "$equipment_manifest" \
    --config "$equipment_config" \
    --output "$equipment_run" \
    --device cuda
