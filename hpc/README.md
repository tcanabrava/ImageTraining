# First Iris test

This tests the **data/training/inference pipeline**, not the final research model.
It uses a small supervised U-Net and one supplied animation, with no network calls
during the job. It produces model weights, numeric diagnostics, and visual previews.
The repository has been tested on CPU; an actual Iris/CUDA run remains to be done.

## 1. Prepare data before submitting

From the project root on a network-enabled preparation machine:

```sh
export PYTHONPATH="$PWD/src"
python -m equipment_fitter fetch-lpc --selection configs/lpc-glove.json --output data/lpc-glove
python -m equipment_fitter fetch-lpc --selection configs/lpc-footwear.json --output data/lpc-footwear
python -m equipment_fitter validate data/lpc-glove/dataset.json
python -m equipment_fitter validate data/lpc-footwear/dataset.json
```

Both datasets have already been downloaded in the development workspace. Skip
fetching when the directories exist. Transfer the repository and the prepared
`data/lpc-glove/` and `data/lpc-footwear/` directories to project/scratch storage
accessible from Iris compute nodes. Git intentionally excludes data, so a Git clone
alone does not transfer it. Keep the credit/provenance files with each dataset.

## 2. Run inside your interactive allocation

From the project root, after `salloc` has granted your GPU allocation:

```sh
bash hpc/run.sh
```

The launcher automatically loads `env/release/2023b` and
`ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0`, creates `.venv-iris/` with
`--system-site-packages`, activates it in the job, checks dependencies and a CUDA
forward/backward pass, and launches the glove pilot. Subsequent runs reuse the
same environment. It imports the project from `src/`; no pip installation or
network download happens in the job.

The release module and GPU PyTorch module names were confirmed in the user's
Iris `module avail 2023b` output on 2026-10-29. The environment setup follows the
[ULHPC GPU Python example](https://hpc-docs.uni.lu/services/jupyter/)
and [module environment documentation](https://hpc-docs.uni.lu/environment/modules/).
Availability on your allocation still needs verification. To check setup without
training (no dataset required):

```sh
bash hpc/run.sh --check
```

The checks require Python 3.11+, NumPy, Pillow, and GPU-compatible PyTorch. If a
module is unavailable, inspect `module spider ai/PyTorch` on the compute node and
set `EQUIPMENT_MODULES` to the available software-set and GPU PyTorch module names.
`EQUIPMENT_RESIF_ARCH` defaults to `gpu`. To use a different managed environment
location, set `EQUIPMENT_VENV`. The launcher refuses to reuse an environment made
with a different Python/module stack; choose a new path in that case.

Existing custom setups remain supported: set `EQUIPMENT_PYTHON` to an already
prepared interpreter, or `EQUIPMENT_ENV_SCRIPT` to an absolute shell script that
loads your modules and activates your environment. These overrides bypass the
automatic module/venv setup, but still run the GPU checks. Module loading and
activation happen in a child process and do not change your interactive shell.

For the footwear pilot inside the same allocation:

```sh
bash hpc/run.sh data/lpc-footwear/dataset.json configs/train-smoke.json
```

## 3. Submit without an interactive allocation

From the project root on Iris:

```sh
mkdir -p logs
sbatch hpc/iris-smoke.sbatch
```

The batch script uses the same automatic environment setup. It requests one GPU, seven CPU cores, 16 GB host RAM, and twenty minutes.
These are initial request limits, not a measured Iris runtime. Pass your Slurm
account/QoS flags if your allocation requires them. The tiny model does not require
a 32 GB GPU specifically. GPU memory and timing are measured in the run output.

The glove test has seven training targets and no held-out designs. A successful
run establishes that the pipeline executes and can learn the training example.
It does not establish generalization.

Then run the footwear check, which has 21 training and 14 validation targets:

```sh
sbatch hpc/iris-smoke.sbatch data/lpc-footwear/dataset.json configs/train-smoke.json
```

For a longer glove overfitting diagnostic:

```sh
sbatch hpc/iris-smoke.sbatch data/lpc-glove/dataset.json configs/train-overfit.json
```

## 4. Inspect and return results

Each run writes `runs/iris-JOBID/`; repeated interactive runs append a timestamp
and process ID to preserve earlier results. Batch logs go in `logs/`; interactive
logs appear in your terminal. The launcher prints the output path. Return these artifacts for
the next engineering/research decision:

- `run.json`, `summary.json`, `baselines.json`, and `metrics.jsonl`.
- The job's `.out`/`.err` logs, especially if CUDA/environment setup fails.
- `previews/` alongside `ground-truth-previews/`: PNG rows are body, equipment,
  and composite; GIFs show playback. The matching source `CREDITS.csv` is retained.

`last.pt` contains the checkpoint. `predictions/` contains RGBA equipment and
composite PNGs; `inputs/` contains the single-reference inference inputs. Inference
does not use target equipment frames. Reference frames are displayed in previews
but excluded from train/validation metrics. Binary output alpha is used for this
pixel-art pilot; raw network metrics are recorded separately.

Inspect missing glove pixels, unwanted opacity, misplaced parts, color drift, and
frame-to-frame jitter. Compare the model to `copy_reference` in `baselines.json`.
Validation is only on different footwear designs on the same body and animation;
it is not a test of new bodies, animation families, or independently generated art.

Current cluster guidance: [Iris GPU hardware](https://hpc-docs.uni.lu/systems/iris/compute/)
and [ULHPC Slurm](https://hpc-docs.uni.lu/slurm/). No job has been submitted by this
project, and no HPC credentials are required for preparing the local pipeline.

Launcher tests simulate Slurm/modules to check setup, reuse, overrides, and failure
handling. They do not replace running `--check` on Iris.
