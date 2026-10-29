"""Exercise the actual shell control flow with mocked Slurm/Lmod/Python commands.

These tests cannot establish that a module exists on Iris or CUDA works there.
"""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def cluster(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    shutil.copytree(Path(__file__).resolve().parents[1] / "hpc", root / "hpc")
    for name in ("data/lpc-glove/dataset.json", "configs/train-smoke.json", "pyproject.toml"):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    (root / "src/equipment_fitter").mkdir(parents=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "commands.log"
    init = tmp_path / "bash-env.sh"
    init.write_text('''module() {
  printf 'module:%s\\n' "$*" >> "$TEST_LOG"
  if [[ "$1" == "--force" ]]; then export LOADEDMODULES=""; fi
  if [[ "$1" == "load" ]]; then
    if [[ "${TEST_MODULE_FAIL:-0}" == "1" ]]; then return 1; fi
    export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$2"
  fi
  return 0
}
''')
    (binaries / "srun").write_text('''#!/usr/bin/env bash
printf 'srun:%s\\n' "$*" >> "$TEST_LOG"
while [[ "$1" != "bash" ]]; do shift; done
exec "$@"
''')
    mock_python = binaries / "python"
    mock_python.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, shlex, shutil, sys
args = sys.argv[1:]
with open(os.environ["TEST_LOG"], "a") as f:
    f.write("python:" + json.dumps({"args": args, "venv": os.environ.get("VIRTUAL_ENV"), "executable": sys.argv[0]}) + "\\n")
if args[0] == "-c":
    print("/modules/python/bin/python\\n3.11.8")
elif args[:2] == ["-m", "venv"]:
    assert "--system-site-packages" in args
    venv = pathlib.Path(args[-1])
    (venv / "bin").mkdir(parents=True)
    shutil.copyfile(__file__, venv / "bin/python")
    (venv / "bin/python").chmod(0o755)
    (venv / "bin/activate").write_text("export VIRTUAL_ENV=" + shlex.quote(str(venv)) + "\\nexport PATH=" + shlex.quote(str(venv / "bin")) + ':"$PATH"\\n')
elif args[0].endswith("preflight.py"):
    if os.environ.get("TEST_PREFLIGHT_FAIL") == "1":
        print("Mock CUDA failure", file=sys.stderr)
        sys.exit(7)
elif args[:3] == ["-m", "equipment_fitter", "pilot"]:
    out = pathlib.Path(args[args.index("--output") + 1])
    out.mkdir(parents=True)
    (out / "mock-success").write_text("test")
else:
    sys.exit(9)
''')
    for p in binaries.iterdir():
        p.chmod(0o755)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("EQUIPMENT_", "SLURM_")):
            del env[name]
    env.update(PATH=str(binaries) + os.pathsep + env["PATH"], BASH_ENV=str(init),
               TEST_LOG=str(log), SLURM_JOB_ID="12345", SLURM_CPUS_PER_TASK="7")
    return root, log, env, mock_python


def launch(cluster, *args, changes=None, batch=False):
    root, _, env, _ = cluster
    env = {**env, **(changes or {})}
    script = "iris-smoke.sbatch" if batch else "run.sh"
    if batch:
        env["SLURM_SUBMIT_DIR"] = str(root)
    return subprocess.run(["bash", str(root / "hpc" / script), *args], env=env,
                          cwd=root.parent, text=True, capture_output=True, timeout=20)


def python_events(cluster):
    return [json.loads(line.removeprefix("python:")) for line in cluster[1].read_text().splitlines()
            if line.startswith("python:")]


def test_bootstraps_activates_checks_and_launches_then_reuses(cluster):
    root, log, _, _ = cluster
    result = launch(cluster)
    assert result.returncode == 0, result.stderr
    text = log.read_text()
    assert "module:load env/development/2023b" in text
    assert "module:load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0" in text
    events = python_events(cluster)
    check, run = events[-2:]
    assert check["args"][0].endswith("preflight.py")
    assert run["args"][:3] == ["-m", "equipment_fitter", "pilot"]
    assert run["args"][-2:] == ["--device", "cuda"]
    assert check["venv"] == run["venv"] == str(root / ".venv-iris")
    assert run["executable"] == str(root / ".venv-iris/bin/python")
    assert (root / "runs/iris-12345/mock-success").is_file()
    result = launch(cluster)
    assert result.returncode == 0, result.stderr
    assert len([x for x in python_events(cluster) if x["args"][:2] == ["-m", "venv"]]) == 1
    assert len(list((root / "runs").glob("iris-12345*"))) == 2


def test_batch_uses_same_setup(cluster):
    result = launch(cluster, batch=True)
    assert result.returncode == 0, result.stderr
    assert python_events(cluster)[-1]["venv"].endswith(".venv-iris")


def test_check_does_not_train_or_require_dataset(cluster):
    (cluster[0] / "data/lpc-glove/dataset.json").unlink()
    result = launch(cluster, "--check")
    assert result.returncode == 0, result.stderr
    assert "check passed" in result.stdout
    assert not any(e["args"][:2] == ["-m", "equipment_fitter"] for e in python_events(cluster))


@pytest.mark.parametrize("changes, message", [
    ({"TEST_MODULE_FAIL": "1"}, "module spider"),
    ({"TEST_PREFLIGHT_FAIL": "1"}, "Mock CUDA failure"),
])
def test_setup_failures_prevent_training(cluster, changes, message):
    result = launch(cluster, changes=changes)
    assert result.returncode != 0
    assert message in result.stderr
    assert not any(e["args"][:2] == ["-m", "equipment_fitter"] for e in python_events(cluster))


def test_changed_stack_does_not_reuse_environment(cluster):
    assert launch(cluster, "--check").returncode == 0
    stamp = (cluster[0] / ".venv-iris/.equipment-runtime").read_bytes()
    result = launch(cluster, changes={"EQUIPMENT_MODULES": "env/other ai/PyTorch/other"})
    assert result.returncode != 0
    assert "differ" in result.stderr
    assert (cluster[0] / ".venv-iris/.equipment-runtime").read_bytes() == stamp


def test_custom_interpreter_is_respected(cluster):
    result = launch(cluster, changes={"EQUIPMENT_PYTHON": str(cluster[3])})
    assert result.returncode == 0, result.stderr
    assert "module:" not in cluster[1].read_text()
    assert not (cluster[0] / ".venv-iris").exists()


def test_custom_environment_script_is_sourced(cluster):
    script = cluster[0] / "custom.sh"
    script.write_text("export EQUIPMENT_PYTHON=" + shlex.quote(str(cluster[3])) + "\n")
    result = launch(cluster, changes={"EQUIPMENT_ENV_SCRIPT": str(script)})
    assert result.returncode == 0, result.stderr
    assert "module:" not in cluster[1].read_text()


def test_no_allocation_fails_before_dispatch(cluster):
    result = launch(cluster, changes={"SLURM_JOB_ID": ""})
    assert result.returncode != 0
    assert "salloc" in result.stderr
    assert not cluster[1].exists()


def test_missing_dataset_fails_before_environment_changes(cluster):
    result = launch(cluster, "data/not-transferred.json")
    assert result.returncode != 0
    assert "Dataset/config missing" in result.stderr
    assert "module:" not in cluster[1].read_text()
