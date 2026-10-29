# Initial CPU pipeline results

Recorded 2026-10-29. These are single-seed engineering smoke tests, not results for
the final research architecture. Both use `configs/train-smoke.json`: 200 updates,
batch size 8, seed 42, width 16, float32, and four CPU threads. The model has
120,548 parameters. Environment: Python 3.14.8, PyTorch 2.14.1, NumPy 2.5.4,
Pillow 13.0.0. CUDA was unavailable locally; no GPU performance is measured.

## Glove: training-set diagnostic

One equipment design, seven non-reference training frames, no validation designs.

| Output | Alpha IoU ↑ | Visible RGB MAE ↓ | Premultiplied MAE ↓ |
| --- | ---: | ---: | ---: |
| Empty overlay | 0.0000 | 0.4846 | 0.01168 |
| Copy reference to every frame | 0.6250 | 0.2459 | 0.00761 |
| Model after 200 steps | 0.9871 | 0.0269 | 0.00601 |

Training loss decreased from 2.5202 to 0.3599. The run completed its training and
evaluation loop in about 27 seconds locally. That is not an Iris runtime estimate.
The exported equipment follows the supplied hand positions in this training example.
The inference command ran using only the base sheet and single reference pair.

Artifacts: `runs/lpc-glove-cpu-smoke/`, `runs/glove-cpu-prediction/`.
Dataset fingerprint:
`c12986bb29715279ad578e37b9b0876677ffe4c6f8ed69c8f514497a76adea35`.

## Footwear: tiny held-out-design check

Three training designs (21 target frames) and two held-out designs (14 frames), all
using the same male character and south-facing walk.

| Output on held-out designs | Alpha IoU ↑ | Visible RGB MAE ↓ | Premultiplied MAE ↓ |
| --- | ---: | ---: | ---: |
| Empty overlay | 0.0000 | 0.3273 | 0.00727 |
| Copy reference to every frame | 0.5934 | 0.2263 | 0.00654 |
| Model after 200 steps | 0.7464 | 0.2038 | 0.00992 |

Final training alpha IoU was 0.9735, substantially above held-out IoU. The model
improves held-out silhouette overlap and visible RGB error relative to copying the
reference, but its whole-frame premultiplied error is worse than both baselines.
These metrics use raw predicted soft alpha; exported pixel-art previews threshold
alpha at 0.5 and are not the exact images underlying the soft-alpha error metric.

Visual inspection confirms important failures: plate boots lose the dark metallic
appearance, and ghillies have stray/missing pixels across frames. This is not
production-quality design preservation. Validation IoU also declined between step
150 and step 200; do not infer that more training alone will fix the model.

The full train/evaluation loop took about 78 seconds on the local CPU, followed by
standalone predictions and GIF/contact-sheet export for all five designs. CPU
timings are noisy and are not a cross-model speed comparison.

Artifacts: `runs/lpc-footwear-cpu-smoke/`, including `previews/` and
`ground-truth-previews/`.
Dataset fingerprint:
`c5325d57aa8f3f0686df1082724c341a803a91f74f93121361dbf95edbf995ea`.

## Verification and next decision

- Twelve automated tests passed, covering dataset leakage/geometry, provenance
  import, alpha-aware loss, gradients, CPU checkpoint-resume equivalence, inference
  without target sheets, and offline pilot artifact generation.
- The Slurm launcher passes `bash -n`. Actual Slurm scheduling, CUDA kernels, and
  V100 compatibility require the user's Iris run.
- Run metadata records development snapshots with `git_dirty: true`; the glove run
  preceded the baseline commit and the footwear run included the then-uncommitted
  pilot wrapper. The completed source is committed separately from ignored run/data
  artifacts. These runs are development measurements, not clean-release benchmarks.

Next: run [the Iris smoke job](../hpc/README.md), then evaluate candidate RGBA
representations and independent-reference conditioning as described in the
[literature review](LITERATURE_REVIEW.md). No independently generated equipment
illustrations, new bodies, behind-torso occlusions, or flexible equipment have yet
been demonstrated.
