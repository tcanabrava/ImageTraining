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


## Frozen reserved-design test (Iris job 6257931)

The [protocol](HELDOUT_PROTOCOL.md) and exact checkpoint/dataset hashes were
committed as `d62dd34` before evaluation. All six final checkpoints from job
6257841 were scored without retraining, seed selection, palette snapping, or
threshold changes. Iris completed inference and export in 17 seconds, exit code 0.
There are 56 scored frames: slippers and tabi socks, four views, with one aligned
reference per design/view excluded from scoring. Same body and walk as development.

| Seed | U-Net foreground MAE ↓ | Transfer foreground MAE ↓ | U-Net IoU ↑ | Transfer IoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.18508 | 0.11146 | 0.6755 | 0.7259 |
| 43 | 0.20006 | 0.10986 | 0.6977 | 0.7311 |
| 44 | 0.17303 | 0.11896 | 0.7036 | 0.7191 |
| Mean | 0.18606 | 0.11343 | 0.6923 | 0.7254 |

These are actual PNG export scores (binary alpha, 8-bit color). Mean foreground
composite MAE decreases by 39.0% and IoU improves in every seed. Mean alpha recall
is 0.8324 versus 0.8469; mean whole-frame premultiplied MAE is 0.004994 versus
0.002898. Copy-reference scores foreground MAE 0.17663, IoU 0.4020, and
premultiplied MAE 0.005443: U-Net's mean foreground score is worse than this simple
baseline, while transfer improves it. Do not substitute whole-frame error for
small-equipment detail quality.

Per-family means below weight each view and seed equally (not the pooled pixel
weighting above):

| Family | U-Net foreground MAE | Transfer foreground MAE | U-Net IoU | Transfer IoU |
| --- | ---: | ---: | ---: | ---: |
| Slippers | 0.21244 | 0.21005 | 0.4495 | 0.4950 |
| Tabi socks | 0.16780 | 0.05760 | 0.8701 | 0.8779 |

The appearance gain is overwhelmingly on tabi socks; slippers remain weak.
Equal-design/seed foreground means by direction are north 0.14604→0.08932,
west 0.20931→0.16613, south 0.19451→0.12428, east 0.21063→0.15557.
These are descriptive breakdowns on just two new families, not confidence bounds.

As predeclared, seed 42 previews from both architectures were inspected against
ground truth: south slippers and west tabi socks. Transfer retains the tabi brown
color where U-Net drifts to dark gray, but shape/detail mismatches persist. Slippers
show misplaced or missing pixels and incorrect coverage, despite the aggregate
improvement. Contact sheets do not establish temporal consistency or production
quality. No independent drawings, new bodies, new movement families, or broader
equipment categories have been tested.

All predictions, input exports, previews, reports, and logs are retained on Iris
and copied locally under `runs/heldout-6257931/` and `logs/heldout-6257931.*`.
Versioned [evidence](benchmarks/heldout-6257931/) includes the full scores, frozen
protocol, and per-candidate environment records. The copied protocol hash matches
the precommitted file, and local source checkpoints still match their frozen hashes.
Remote user launcher edits remain preserved and recorded as dirty in inference
metadata. The freeze identifies exact weights independently of working-tree status.

Validation: 32 tests pass, including frozen hash/fingerprint rejection, disjoint
family validation, reference exclusion, and inference leaving checkpoints unchanged.
The Slurm script passed ShellCheck; all six GPU evaluations completed.

Decision: explicit reference color transfer is supported as a useful constrained
mechanism on these two test families, with a serious remaining geometry limitation.
Keep it experimental. Slippers/tabi are now a consumed test set and must not be
claimed as untouched evidence for any future changes informed by these results.
Next define a new benchmark with independently prepared references and broader
attachment/occlusion cases before tuning or extending the model toward the product.


## Reference registration stress (Iris job 6257989)

Protocol and implementation committed as `fc53f7a` before evaluation. All six
expanded-footwear checkpoints remained frozen. This experiment used only the
eight existing development validation samples (56 scored frames); no consumed
test families were rescored. Iris completed all 24 candidate/condition evaluations
in 29 seconds, exit code 0. Full [protocol](REFERENCE_STRESS.md) and
[per-seed/design evidence](benchmarks/reference-stress-6257989/reference-stress.json)
are versioned.

| Condition | U-Net IoU ↑ | Transfer IoU ↑ | U-Net foreground MAE ↓ | Transfer foreground MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Aligned control | 0.7404 | 0.7795 | 0.16564 | 0.12925 |
| Body and item shifted together | 0.6602 | 0.6855 | 0.17593 | 0.13853 |
| Item shifted, original body | 0.5203 | 0.4646 | 0.23293 | 0.19860 |
| Centered item, transparent body | 0.1744 | 0.1931 | 0.21162 | 0.22646 |

Means include all three seeds. The aligned condition matches every previous
expanded-development metric within 1e-6, validating the control path. All 32
transformed reference pairs preserve equipment pixels without clipping/rescaling.
Each transformed reference and its body input are saved as separate PNGs with
source hash and explicit `independently_drawn: false` provenance.

The pair-shift result exposes sensitivity to absolute reference coordinates even
when mutual body/equipment alignment is preserved. Shifting only the item causes
larger failures; centering it and removing the reference body is catastrophic.
The last two conditions deliberately violate the trained aligned-input contract.
They establish the current contract's limitation, not the performance of a model
trained for independent references. The centered condition changes two factors at
once and cannot isolate their individual causal effects.

Foreground error alone understates the spurious overlay problem: transfer's
whole-frame premultiplied MAE rises from 0.003331 (aligned) to 0.020821 (centered
without body); U-Net rises from 0.004807 to 0.032959. Transfer seed-42 south plate
boot contact sheets show right-shifted fragments with item-only translation and
large unwanted torso overlays when the reference is centered without a body.
Source body pixels are not regenerated, but false opaque equipment still hides
them in the composite. Thus preserving source pixels is not sufficient to ensure
usable visual output.

No new drawings or independently prepared designs were evaluated. This is a
registration stress test derived exclusively from the original reference frames,
not independent-reference capability. Transformed inputs, predictions, previews,
reports and logs are retained on Iris and locally under
`runs/reference-stress-6257989/` and `logs/reference-stress-6257989.*`.

Validation: 34 tests pass; transform checks cover visible-color preservation,
source immutability, centered placement, and clipping/empty-reference rejection.
All actual development reference transforms also passed the no-clipping check.
The Slurm entry point passed ShellCheck and completed successfully on GPU.

Decision: do not expose the current fitting model as accepting arbitrary standalone
art. Next define a training input contract with canonical item references and
explicit attachment information, then test registration augmentation and removal
of the reference-body dependency separately on development data. Independently
authored paired references and new held-out families are still required afterward;
broader equipment and occlusion remain separate research gates.


## Canonical references and attachment regions (Iris job 6258458)

Implementation and [protocol](CANONICAL_REFERENCE.md) committed as `e89b2f0` before
training. Six new models were trained on the expanded development set only:
canonical reference transfer, with `none` or `feet` attachment regions, seeds
42/43/44, 1,600 updates, batch 8. No consumed test-set evaluation occurred.
Iris completed the job in 5 minutes 33 seconds, exit code 0.

Canonical input centers the visible item bounding box and supplies no reference
body. The reference still contains paired equipment from one original pose/view;
this is not an independently drawn item. The optional foot region is a broad
rectangle computed from the lower third of target-body alpha bounds. It is both
an input and an output alpha restriction. A pretraining coverage audit found all
development target pixels inside the fixed region; the rule was not tuned on
those targets. That coverage does not establish suitability for new equipment.

| Seed | No-region foreground MAE ↓ | Foot-region foreground MAE ↓ | No-region IoU ↑ | Foot-region IoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.12369 | 0.12835 | 0.7560 | 0.7664 |
| 43 | 0.12806 | 0.12407 | 0.7624 | 0.7565 |
| 44 | 0.12116 | 0.12365 | 0.7682 | 0.7554 |
| Mean | 0.12430 | 0.12535 | 0.7622 | 0.7595 |

Scores use final PNG-equivalent outputs on the same 56 validation targets. Mean
recall is 0.9476 (none) versus 0.9418 (feet); whole-frame premultiplied error is
0.003761 versus 0.003529. The region reduces whole-frame error but does not improve
the primary foreground score or IoU consistently. There is no evidence here to
prefer it as the default on quality grounds.

For context, frozen aligned-reference transfer scored foreground MAE 0.12925,
IoU 0.7795 on this development set. Canonical training therefore reaches similar
foreground error without a reference body, with lower silhouette overlap. The old
aligned model's centered/no-body stress score was IoU 0.1931. Recovery to ~0.76
requires the new training contract; it is not evidence that the old model became
robust. Canonicalization, key features, and training inputs change together, so the
comparison does not isolate one causal architectural improvement.

Seed-42 plate-boots-south and ghillies-west contact sheets were inspected for both
variants. There are no large torso overlays in those examples, but boot detail is
smoothed and ghillies remain overly filled/dark with missing strap structure.
Independent art, single-shoe references, scale/rotation changes, other equipment
categories, occlusion, and new bodies remain untested.

Standalone CPU inference with each seed-42 variant succeeded with the reference
body file absent. Shifting the item eight pixels produced byte-identical equipment
PNGs after centering. This translation invariance is built into preprocessing and
is not a learned generalization claim. Outputs are in
`runs/canonical-standalone-check/`. Canonical checkpoints use the explicit
`canonical-footwear-rgba-v1` contract; old checkpoints retain their aligned contract.

Training/evaluation loops took 40.59–43.41 seconds; peak allocated CUDA memory was
158.0 MiB in the shared sequential process. Six checkpoint hashes were verified
against their diagnostic reports after copying locally. All artifacts are retained
on Iris and locally in `runs/canonical-6258458/`, with Slurm logs in `logs/`.
Versioned [evidence](benchmarks/canonical-6258458/) retains all seeds and metrics.

Validation: 37 tests passed, including canonical centering, attachment gating,
resume/inference for the new architecture, and complete pilot export without a
reference-body file. The Slurm script passed ShellCheck and all GPU runs completed.

Decision: retain canonical/no-region as a viable experimental input contract;
retain the region variant as a measured ablation, not a demonstrated quality gain.
Next test shape/scale and independently prepared references under this contract,
with new evaluation families before making further generalization claims. Do not
reuse the consumed slippers/tabi test set to validate these new models.
