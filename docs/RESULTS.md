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


## Export-aware appearance ablation (Iris job 6257139)

Revision `855ab11`, completed on V100 with exit code 0 in 8 seconds. Reused the
footwear checkpoint from job `6257117`; no retraining or new model selection.
The diagnostic excludes reference frames. Full per-design scores and checkpoint
SHA-256 are in [the report](benchmarks/appearance-6257139/appearance.json).
The matching local CPU diagnostic agreed to displayed precision.

| Output, held-out frames | Binary-alpha IoU ↑ | Export premultiplied MAE ↓ | Target-foreground composite MAE ↓ |
| --- | ---: | ---: | ---: |
| Copy reference | 0.5934 | 0.006544 | 0.22632 |
| Existing model, PNG-equivalent export | 0.7020 | 0.005479 | 0.22202 |
| Model + nearest reference palette | 0.7020 | 0.005189 | 0.20948 |

**Metric clarification:** the earlier 0.00989 model premultiplied error measures
soft alpha before export. With binary alpha and 8-bit rounding, the actual export
beats copy-reference on this metric. This does not remove its visible design and
missing-pixel failures. Future comparisons must state which output they score.
Foreground composite error explicitly penalizes equipment that disappears.

Palette projection improves pooled foreground composite MAE by 5.65%, without
changing geometry. Plate-boots error improves from 0.19044 to 0.17247; ghillies
only from 0.27900 to 0.27625. Ghillies recall is 0.5969: about 40.3% of true
foreground pixels remain missing. Even projected plate-boots foreground error is
worse than copy-reference (0.16797), despite better silhouette overlap.

An oracle using true target geometry and assigning each target color its nearest
reference palette color has zero error on these held-out frames. Thus this small
subset's reference palette contains all necessary target colors. This is a
representation diagnostic using ground truth, not inference performance or proof
that other equipment/animation palettes will be sufficient.

Inverting reference RGB while holding reference alpha, bodies, and coordinates
fixed changes output RGB by 0.1724 for plate boots and 0.0817 for ghillies, averaged
on target-visible pixels. Therefore the network does respond to reference color.
It does not reliably reproduce that transformation: inversion-equivariance errors
are 0.4181 and 0.3778. Inverted references are outside the training distribution;
this probe alone cannot establish whether a particular internal path is ignored.

Decision: keep palette projection as a diagnostic, not the production solution.
Next compare explicit reference-feature transfer against the concatenation
baseline, with the same split, export-aware per-design metrics, and controlled
color augmentation. Evaluate appearance and alpha recall separately. Do not add a
temporal model or select a latent representation based on this palette test.
The trainable RGBA representation gate remains outstanding.

Validation: 24 tests passed, including hidden-RGB exclusion from palettes and
export metric checks; ShellCheck passed for the new Slurm entry point. The GPU
report records the preserved remote chmod/untracked launcher changes as dirty.


## Learned reference transfer: three-seed comparison (Iris job 6257150)

Revision `37ab92e`, V100, Slurm exit code 0, 39 seconds for all six runs and their
exports/diagnostics. Seeds 42, 43, and 44 each trained the original U-Net and the
new reference-transfer model for 200 updates, with identical data, optimizer,
batch size, and loss. Architecture changes initialization and parameter count;
this is not an equal-parameter or equal-runtime comparison. Final checkpoints are
reported without selecting seeds or early-stopping steps on validation.

The experimental model learns target query features and reference-body keys,
attends to visible reference locations, and mixes the original equipment colors.
Alpha and attention weights are independent of reference equipment RGB. Colors
are limited to the reference RGB convex hull (mixtures can introduce intermediate
colors). This is an intentionally constrained project experiment, not a paper
reproduction or a general equipment system.

| Seed | U-Net foreground MAE ↓ | Transfer foreground MAE ↓ | U-Net IoU ↑ | Transfer IoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.22202 | 0.16606 | 0.7020 | 0.7832 |
| 43 | 0.19398 | 0.15644 | 0.8030 | 0.7725 |
| 44 | 0.20525 | 0.16148 | 0.8159 | 0.7789 |
| Mean | 0.20708 | 0.16133 | 0.7736 | 0.7782 |

Scores use PNG-equivalent binary alpha and quantization on the same 14 held-out
frames. Transfer improves pooled foreground composite MAE in each seed, by 22.1%
on the mean. Mean whole-frame premultiplied error improves from 0.005521 to
0.004631, and mean alpha recall from 0.8790 to 0.9340. IoU declines in two seeds,
so this is not uniform geometry improvement. Higher recall can include excessive
coverage; per-design IoU and previews remain essential.

Plate-boots foreground MAE improves in every seed: 0.19044→0.12007,
0.17769→0.11298, 0.19199→0.13297. Ghillies is mixed: 0.27900→0.24902,
0.22337→0.23484, 0.22918→0.21292. The difficult thin details are not solved.
Contact sheets inspected for transfer seeds 42 and 44 show darker plate boots
closer to the reference but softened internal detail. Ghillies become broad,
filled shoe-like shapes rather than faithful straps. These observations do not
establish temporal stability.

The U-Net has 120,548 parameters; transfer has 124,033 (+2.9%). Training/evaluation
loops took 1.995–2.661 seconds for U-Net and 4.186–4.298 seconds for transfer.
Recorded peak allocated memory was 147.1–157.1 MiB and 158.9 MiB respectively.
These sequential runs share a process and allocator; they are not isolated serving
memory benchmarks. Model-loading and preview time are excluded from loop timings.

All runs, checkpoints, and previews are copied locally and retained on Iris in
`runs/compare-6257150/`. Versioned [comparison evidence](benchmarks/compare-6257150/)
contains all seeds, per-design diagnostics, learning curves, environment metadata,
and checkpoint hashes. Remote user chmod/untracked launcher changes remain intact
and are recorded as dirty; experiment source was committed before running.

Validation: 27 tests passed, including finite transfer gradients, color-inversion
behavior, architecture-compatible resume, and inference without ground-truth
sheets for both architectures. The Slurm script passed ShellCheck and the full
six-run GPU experiment completed successfully.

Decision: retain reference transfer as a promising experimental variant; leave
U-Net as the default and preserve both baselines. Next expand training and an
untouched evaluation set with thin/structured equipment and additional animation
views before tuning correspondence or detail losses. Repeated development on
these two validation designs is not an unbiased final generalization test.
Independent item illustrations, behind-torso occlusion, new bodies, and latent
RGBA representation fidelity remain unvalidated.


## Expanded four-view development benchmark (Iris job 6257841)

Dataset/config and split safeguards were committed as `e74ea59` before execution.
The earlier commit attempt was delayed by an approval-service usage limit; the
retry succeeded without changing the approval mechanism. All six GPU runs then
completed successfully on Iris in 4 minutes 20 seconds, Slurm exit code 0.

The expanded data has 168 training targets and 56 validation targets across
north/west/south/east views. The development fingerprint is
`2daacc0965ecb1236368b1cc988a08e73f79ebfc866ef9bfff1336b76aa1f75e`.
Training adds sandals and revised boots/shoes. Each direction has its own aligned
reference; all directions occur in training. This is neither unseen-view nor
independent-illustration testing. The 56 reserved slippers/tabi test frames were
not used for training, model evaluation, or checkpoint selection.

Both architectures trained for a predeclared 1,600 updates, batch size 8, seeds
42/43/44. This scales updates eightfold with the eightfold training-frame increase.
It does not isolate the effect of new data versus longer optimization; compare
architectures within this experiment, not scores against the earlier dataset.

| Seed | U-Net foreground MAE ↓ | Transfer foreground MAE ↓ | U-Net IoU ↑ | Transfer IoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.16850 | 0.13358 | 0.7277 | 0.7775 |
| 43 | 0.17142 | 0.12470 | 0.7391 | 0.7912 |
| 44 | 0.15700 | 0.12946 | 0.7544 | 0.7698 |
| Mean | 0.16564 | 0.12925 | 0.7404 | 0.7795 |

All scores use final-checkpoint, PNG-equivalent outputs. Transfer lowers pooled
foreground composite error by 22.0% on the mean and improves silhouette IoU in all
three seeds. Mean alpha recall is 0.8940 versus 0.9179; mean whole-frame
premultiplied MAE is 0.004807 versus 0.003331. Three seeds and two validation
families are insufficient for a broad generalization claim.

Direction breakdown below averages foreground MAE equally over the two validation
designs and three seeds, unlike the pixel-weighted pooled scores above:

| View | U-Net | Transfer |
| --- | ---: | ---: |
| North | 0.15880 | 0.08591 |
| West | 0.16637 | 0.12767 |
| South | 0.19629 | 0.19117 |
| East | 0.16783 | 0.12781 |

The small south-view gain is a remaining weakness hidden by the pooled mean.
Equally averaged across views/seeds, ghillies improve from 0.20333 to 0.16617 and
plate boots from 0.14132 to 0.10011. Contact sheets inspected for transfer seed 42:
west ghillies preserve some straps but still misplace/fill pixels; north plate
boots show extra or misplaced shapes in several frames. Metrics are encouraging,
not pixel-perfect fidelity or evidence of temporal consistency.

Training/evaluation loops took 17.76–19.40 seconds for U-Net and 39.99–42.13 seconds
for transfer. Sequential-process allocated CUDA peaks were at most 155.2 MiB and
157.0 MiB; these are not isolated serving benchmarks. Full artifacts are retained
on Iris and copied locally under `runs/compare-6257841/`; all six checkpoint
hashes match the saved diagnostic reports. Compact evidence is versioned in
[compare-6257841/](benchmarks/compare-6257841/), with all seeds and per-design scores.

Validation: 29 tests passed before submission, including multi-view family split
checks and refusal to run comparisons on manifests containing reserved tests.
Source assets retain pinned hashes and credits. Exact equipment-file duplicates
across splits were checked and none found. Selected data previews were inspected;
this is not an exhaustive occlusion or ancestry audit. User remote launcher edits
remain preserved and recorded as dirty in run metadata.

Next: freeze the two candidate models and the evaluation protocol before scoring
reserved slippers/tabi designs. Report all three seeds without choosing the best
one, and include per-design/view failures. Do not tune on that test result; new
model iterations will require a new untouched evaluation set.
