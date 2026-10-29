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


## Iris GPU benchmark (2026-10-29)

Both pilots now ran successfully on a Tesla V100-SXM2-16GB (compute capability
7.0), Python 3.11.5, PyTorch 2.3.0/CUDA 12.6, NumPy 1.26.2, Pillow 10.2.0.
The 200-step configuration, seed, and dataset fingerprints match the CPU tests.
CPU/GPU results are not numerically identical; package versions also differ.

| Run | Evaluated split | Alpha IoU ↑ | Visible RGB MAE ↓ | Premultiplied MAE ↓ |
| --- | --- | ---: | ---: | ---: |
| Glove `6257007` | Training, 7 frames | 0.9871 | 0.02174 | 0.00612 |
| Footwear `6257117` | Training, 21 frames | 0.9717 | 0.06965 | 0.00844 |
| Footwear `6257117` | Held-out, 14 frames | 0.7020 | 0.19979 | 0.00989 |
| Copy reference | Held-out, 14 frames | 0.5934 | 0.22632 | 0.00654 |
| Empty overlay | Held-out, 14 frames | 0.0000 | 0.32732 | 0.00727 |

Glove train/evaluation time was 2.264 s, with 2.059 s in optimization and 777
training examples/s. Footwear train/evaluation time was 2.631 s, with 2.324 s in
optimization and 688 training examples/s. Each run peaked at 147.1 MiB allocated
and 202 MiB reserved CUDA memory. These timings exclude module setup, GPU
preflight, exports, and queueing; the footwear Slurm job completed in 15 seconds,
exit code 0. Tiny workloads and different software versions make these unsuitable
for a general CPU-versus-GPU speedup claim.

Contact sheets inspected locally against ground truth: glove placement follows the
provided hand positions closely, with fine shading differences. Held-out plate
boots become pale and lose metallic detail. Ghillies lose substantial equipment
pixels, including nearly disappearing feet in some frames. This confirms the
appearance-preservation problem observed on CPU. The raw soft-alpha metrics above
and thresholded preview exports measure different outputs, as explained earlier.
No temporal-quality claim follows from contact-sheet inspection alone.

The glove run used revision `7770fcf`; footwear used `51830bd`. Both record
`git_dirty: true`: inspection found a chmod-only change to `hpc/run.sh` and an
untracked user `run_salloc.sh`, which were preserved. Training source was unchanged.
The first automated batch attempt, `6257111`, failed before training because
`ULHPC_CLUSTER` was absent in the noninteractive submission environment. Revision
`51830bd` supplies the Iris default; retry `6257117` completed successfully.

Full checkpoints and previews are retained on Iris and copied locally under
`runs/iris-6257007/` and `runs/iris-6257117/`. Batch logs are copied under
`logs/iris-benchmark/`. These large artifacts remain Git-ignored. Compact run
metadata, baseline scores, summaries, and learning curves are versioned in
[benchmarks/](benchmarks/). Ten launcher regression tests and ShellCheck passed
for the batch-environment fix, followed by actual successful Slurm execution.

**Next experiment:** measure exported binary-alpha/color errors explicitly and
compare reference-appearance conditioning under the same held-out split before
increasing model size or training duration. Preserve these runs as the baseline.
The RGBA reconstruction gate from the literature review still applies before
selecting any latent generative model. Independent illustrations, unseen bodies,
other animation families, and flexible equipment remain untested.
