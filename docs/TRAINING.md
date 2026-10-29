# Training and inference baseline

The first model is a small conditional U-Net, trained from scratch as a fitting
baseline. It takes target body RGBA, reference body RGBA, reference equipment RGBA,
and XY coordinates. RGB inputs are premultiplied by alpha. It predicts equipment
RGB/alpha only. This is not a diffusion model, a design generator, or a temporal model.

Use Python 3.11+ with NumPy, Pillow, and PyTorch 2.2+. Select the PyTorch build for
your hardware before installing the package; the optional `train` extra is a
convenience, not a V100-compatible environment lock. Exact versions are recorded
in every run. Tests also require pytest.

```sh
export PYTHONPATH="$PWD/src"
python -m pytest -q
python -m equipment_fitter train \
  --manifest data/lpc-glove/dataset.json \
  --config configs/train-smoke.json \
  --output runs/glove-smoke --device cpu
```

Use `--device cuda` on an allocated GPU. It fails if CUDA is unavailable instead of
silently falling back to CPU. The 200-step smoke config checks the pipeline;
`configs/train-overfit.json` runs 1,000 steps for the single-design diagnostic.
Neither is a convergence guarantee or a GPU-time estimate.

The training sampler draws a seeded batch with replacement each step. The small
pilot is cached in host memory. Batch size therefore need not be smaller than the
number of frames. The loss combines foreground-balanced alpha BCE, Dice loss,
visible RGB L1, and balanced premultiplied RGB error. RGB below zero target alpha
does not affect the loss. Checkpoints are saved atomically at logging intervals.

Outputs:

- `run.json`: configuration, code revision/dirty state, package versions, device,
  parameter count, dataset fingerprint, and resume provenance.
- `dataset.json`: exact training manifest (image paths refer to the original dataset).
- `metrics.jsonl`: loss and full train/validation metrics at logging intervals.
- `baselines.json`: empty-overlay and unchanged-reference comparisons.
- `last.pt`: model, optimizer, step, dimensions, and input contract.
- `summary.json`: final metrics, training throughput, and GPU peak memory if available.

Alpha IoU uses threshold 0.5; visible RGB MAE is measured on ground-truth foreground;
premultiplied MAE is averaged across the whole frame. Lower whole-frame error can
favor empty output for tiny items, so inspect alpha IoU and the visual previews.
Training-set glove metrics demonstrate fitting capacity only. `val: null` means
there are no validation designs, not a successful validation result.

## Resume

Use a new output directory and a config with a larger total `steps` value:

```sh
python -m equipment_fitter train \
  --manifest data/lpc-glove/dataset.json --config configs/train-overfit.json \
  --resume runs/glove-smoke/last.pt --output runs/glove-continued --device cuda
```

The dataset fingerprint, model width, batch size, learning rate, and seed must
match. Sampling continues from the stored step. CPU resume equivalence is tested;
bitwise results across different GPU/software stacks are not promised.

## Standalone inference

Extract a representative input for testing:

```sh
python -m equipment_fitter export-input \
  --manifest data/lpc-glove/dataset.json --sample metal-gloves --output data/glove-input
python -m equipment_fitter infer \
  --checkpoint runs/glove-smoke/last.pt \
  --base data/glove-input/base.png \
  --reference data/glove-input/reference.png \
  --reference-base data/glove-input/reference-base.png \
  --layout data/glove-input/layout.json \
  --output runs/glove-prediction --device cpu
```

Inference needs only the base sheet, layout, and one aligned reference pair. It
does not read the target equipment sheet. Output contains `equipment.png`,
`composite.png`, unchanged layout metadata, and timing information. The base input
is never modified. Default binary alpha fits opaque pixel-art equipment; choose
`--alpha-mode soft` to preserve predicted alpha values. Neither option guarantees
correct occlusion; visual quality remains a model-evaluation question.

References must currently be aligned RGBA frames at the training frame size. An
arbitrary generated glove illustration is **not yet a supported input contract**.
The next research stage must address reference normalization and appearance
conditioning, and train/evaluate with independent illustrations. This baseline
does not automatically infer frame layout or animation timing from a PNG.

## Reference appearance diagnostic

Run an existing checkpoint through the same benchmark without further training:

```sh
PYTHONPATH=src python -m equipment_fitter.appearance \
  --checkpoint runs/iris-6257117/last.pt \
  --manifest data/lpc-footwear/dataset.json \
  --output runs/appearance-check --device cpu
```

On Iris, submit `sbatch hpc/appearance.sbatch CHECKPOINT MANIFEST` from the project
root. This uses the shared module/venv setup and saves `runs/appearance-JOBID/`.

The diagnostic excludes reference frames and reports per-design and pooled split
scores after the same binary-alpha threshold and 8-bit quantization as PNG export.
Target-foreground composite MAE penalizes missing equipment pixels, unlike RGB
error that ignores predicted visibility. Comparisons include unchanged model
output, copy-reference, and nearest visible-reference-palette projection in sRGB.
Projection preserves predicted alpha and cannot repair missing geometry.

A separate **palette oracle** projects target colors onto the reference palette,
using true target geometry. It measures a representation bound; it is not a valid
prediction and must not be presented as model performance. Reference-color
inversion probes output sensitivity while holding bodies and alpha fixed. Because
inverted references are outside this tiny training distribution, weak response is
evidence of limited color control, not proof the entire reference is ignored.
These are project diagnostic hypotheses, not reproduced paper methods.
