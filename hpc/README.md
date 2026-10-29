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

## 2. Select the Python environment

Use Python 3.11+ with NumPy, Pillow, and a **V100-compatible PyTorch build**. A
cluster-provided PyTorch stack is suitable if it meets those requirements. Do not
assume the latest downloadable CUDA wheel supports V100. The job explicitly requests
CUDA and executes GPU operations; it will fail rather than fall back to CPU.

The launcher imports directly from `src/`, so installing the package itself is not
required. It does not guess module names or change your environment. If needed,
set an absolute interpreter path before submitting:

```sh
export EQUIPMENT_PYTHON=/absolute/path/to/your/environment/bin/python
```

If module loading or environment activation must happen on the compute node, put
the appropriate commands in your own shell file and set `EQUIPMENT_ENV_SCRIPT` to
its absolute path. The job sources that file before starting Python. Run compute
checks and training on allocated compute nodes, not login nodes.

## 3. Submit the glove smoke test

From the project root on Iris:

```sh
mkdir -p logs
sbatch hpc/iris-smoke.sbatch
```

The script requests one GPU, four CPU cores, 16 GB host RAM, and twenty minutes.
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

Each job writes `runs/iris-JOBID/` and logs in `logs/`. Return these artifacts for
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
