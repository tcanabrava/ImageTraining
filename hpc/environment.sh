#!/usr/bin/env bash
# Sourced in a child shell on the allocated compute node.

equipment_fail() {
    echo "Equipment environment: $*" >&2
    exit 1
}

equipment_init_modules() {
    if type module >/dev/null 2>&1; then
        return
    fi
    local init_file
    for init_file in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$init_file" ]]; then
            # Site initialization may reference unset variables.
            set +u
            # shellcheck disable=SC1090
            source "$init_file"
            set -u
            if type module >/dev/null 2>&1; then
                return
            fi
        fi
    done
    equipment_fail "Lmod is unavailable on this node. Run inside an Iris GPU allocation."
}

equipment_setup_environment() {
    local root="$1"
    # Preserve the explicit environment overrides supported by the earlier launcher.
    if [[ -n "${EQUIPMENT_ENV_SCRIPT:-}" ]]; then
        [[ -r "$EQUIPMENT_ENV_SCRIPT" ]] || equipment_fail "Cannot read EQUIPMENT_ENV_SCRIPT: $EQUIPMENT_ENV_SCRIPT"
        set +u
        # shellcheck disable=SC1090
        source "$EQUIPMENT_ENV_SCRIPT"
        set -u
        EQUIPMENT_PYTHON="${EQUIPMENT_PYTHON:-$(command -v python)}"
        export EQUIPMENT_PYTHON
        return
    elif [[ -n "${EQUIPMENT_PYTHON:-}" ]]; then
        # An explicitly chosen interpreter is assumed to have its libraries set up.
        export EQUIPMENT_PYTHON
        return
    fi

    equipment_init_modules
    local module_names="${EQUIPMENT_MODULES:-env/development/2023b ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0}"
    local -a modules
    read -r -a modules <<< "$module_names"
    # Interactive allocations can otherwise inherit CPU-oriented module paths.
    export RESIF_ARCH="${EQUIPMENT_RESIF_ARCH:-gpu}"
    module --force purge || equipment_fail "Could not clear the child shell's module stack."
    local name
    for name in "${modules[@]}"; do
        if ! module load "$name"; then
            echo "Could not load $name. Inspect 'module spider ai/PyTorch' on Iris." >&2
            echo "Set EQUIPMENT_MODULES to the available software-set and GPU PyTorch module names." >&2
            exit 1
        fi
    done
    module list

    local bootstrap python_identity runtime_stamp venv lock_fd
    bootstrap="$(command -v python)" || equipment_fail "Loaded modules did not provide Python."
    python_identity="$("$bootstrap" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"; print(sys.executable); print(sys.version.split()[0])')" \
        || equipment_fail "The loaded Python must be version 3.11 or newer."
    runtime_stamp="$(printf '%s\n' "$module_names" "$RESIF_ARCH" "${LOADEDMODULES:-}" "$python_identity")"
    venv="${EQUIPMENT_VENV:-$root/.venv-iris}"
    [[ "$venv" == /* ]] || venv="$root/$venv"
    mkdir -p -- "$(dirname -- "$venv")"
    # Two jobs may initialize the same environment concurrently. Serialize creation.
    exec {lock_fd}>"${venv}.lock"
    flock "$lock_fd" || equipment_fail "Could not lock environment creation."
    if [[ -e "$venv" ]]; then
        [[ -f "$venv/.equipment-runtime" && -f "$venv/bin/activate" && -x "$venv/bin/python" ]] \
            || equipment_fail "Existing $venv is not a complete managed environment. Set EQUIPMENT_VENV to a new path, or use EQUIPMENT_PYTHON for your existing environment."
        [[ "$(cat "$venv/.equipment-runtime")" == "$runtime_stamp" ]] \
            || equipment_fail "Python/modules differ from those used to create $venv. Set EQUIPMENT_VENV to a new path; the existing environment was left unchanged."
    else
        echo "Creating $venv from the loaded module packages."
        "$bootstrap" -m venv --system-site-packages "$venv"
        printf '%s\n' "$runtime_stamp" > "$venv/.equipment-runtime"
    fi
    # shellcheck disable=SC1090,SC1091
    source "$venv/bin/activate"
    EQUIPMENT_PYTHON="$venv/bin/python"
    export EQUIPMENT_PYTHON
    flock -u "$lock_fd"
    exec {lock_fd}>&-
}
