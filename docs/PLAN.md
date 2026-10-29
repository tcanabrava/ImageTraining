# Project plan

## Product contract

1. A user requests an equipment design, such as "draw a glove", and approves it.
2. The user provides an existing spritesheet and its frame layout/timing.
3. The fitting service produces one aligned equipment RGBA spritesheet.
4. A browser composites and plays the base and equipment frames, and offers downloads.

Equipment follows the supplied character's poses. Body pixels and movement are
never regenerated. Occluded equipment pixels have zero alpha. Each overlay is
specific to its base animation. Design generation and equipment fitting can use
different models. Pose or attachment information may be supplied or estimated.

## Milestones and acceptance criteria

Research prerequisite: use [the literature review](LITERATURE_REVIEW.md) to separate
published evidence from project hypotheses. The initial U-Net is an engineering
baseline. Before milestone 6, benchmark RGBA reconstruction, verify compatible
pretrained components, and define independent-reference supervision. The review was
written after initial implementation began; it does not retroactively validate it.

| Step | Deliverable | Acceptance criterion |
| --- | --- | --- |
| 1 | Scope, plan, and research baseline | Versioned project notes and this plan. |
| 2 | Python package, dataset contract, validation, preview | A deterministic local fixture validates; invalid frame layouts and design leakage fail; preview reproduces layer composition. |
| 3 | Reproducible real-asset pilot | Pinned upstream revision, explicit asset selection and licenses, grouped train/validation split, and visual inspection of paired frames. |
| 4 | PyTorch fitting baseline and HPC smoke job | CPU forward/backward and inference pass locally; Iris entry point records environment, loss, validation metrics, checkpoint, and predictions without downloading during the job. |
| 5 | HPC benchmark and data audit | User runs the smoke job; record GPU memory, throughput, loss curve, and visual failures. Demonstrate tiny-set overfitting before interpreting held-out results. |
| 6 | Stronger fitting model | First validate RGBA reconstruction and component compatibility, then compare reference/spatial conditioning against the baseline. Ablate temporal conditioning for detail loss. Select based on equipment-specific held-out quality and serving cost. |
| 7 | Generalization | Test independent reference illustrations, unseen body templates, animation families, styles, and equipment categories. Audit occlusion and flexible-item behavior separately. |
| 8 | Serving pipeline | Versioned standalone inference, queued jobs, output caching, asset storage, and a measured single-GPU serving configuration. |
| 9 | Website | Design approval, spritesheet upload/layout, fitting job status, layered animation preview, and downloads. |

Each completed implementation step receives its own commit, without a co-author
trailer. The first implementation session targets steps 1–4. Steps 5 onward depend
on experimental evidence; public website deployment is not part of the first session.

## Current status (2026-10-29)

- Steps 1–3: implemented and committed. Real glove and footwear subsets are prepared
  locally with pinned sources and credits; contact sheets have been inspected.
- Step 4: CPU training, checkpoint resume, standalone inference, and the complete
  pilot pipeline are tested. The Iris launcher and V100 execution are verified with completed glove and
  held-out footwear pilots. See [HPC instructions](../hpc/README.md).
- Step 5: initial GPU benchmark completed; metrics, learning curves, memory,
  throughput, and visual failures are in [RESULTS.md](RESULTS.md). Broader data
  auditing and generalization remain outstanding.
- Step 6: export-aware diagnostics and a learned reference-transfer variant are
  implemented. A three-seed Iris comparison improves mean foreground error by
  22.1%, with mixed silhouette results and unresolved thin details. The variant
  remains experimental; the final architecture and RGBA representation are not selected.
- Step 7: expanded four-view development data and reserved slippers/tabi test
  families were evaluated under a precommitted hash-frozen protocol. Transfer
  improves mean test foreground error by 39%, mainly on tabi socks; slippers
  geometry remains weak. These test families are now consumed for future model
  development. Independent-reference and broader-category testing remain outstanding.
- Reference-domain gate: a frozen development registration stress test now shows
  severe failure on shifted/centered standalone presentations. Separate reference
  artifacts and provenance are implemented. Canonical-reference training now
  works without a reference body (~0.76 development IoU); a body-derived foot ROI
  shows no consistent primary-score benefit. Independent art, scale/shape changes,
  broader equipment, and new evaluation families remain outstanding.
- External-reference data path: separately authored images and provenance are
  now supported, with pending review dossiers and anti-copy checks. The first
  text-generated lace-cuff candidate was rejected for a design mismatch. No real
  independent-reference dataset or model score exists yet; see EXTERNAL_REFERENCES.md.
- Steps 8–9: not implemented. Representation fidelity, independent references,
  new bodies, and temporal/detail quality remain research gates.

## Data strategy

Inspect a small subset of Universal LPC body and equipment layers first. Preserve
the upstream commit, source paths, SHA-256 hashes, authors, and licenses. Import
only explicitly selected assets. Check that selected equipment is suitable for
direct overlay; do not assume every upstream asset already has the required
occlusion. Keep a reviewable preview beside each prepared dataset.

Use one equipment frame as the initial reference and the remaining animation
frames as targets. Keep related designs and recolors together in one split. Do not
use the reference frame as a prediction target in the first benchmark. A reference
cropped from a sprite is an easier input than an independently generated drawing;
performance on that benchmark does not establish the final product's quality.

Start with one body, one animation, and one equipment category. If glove design
diversity is inadequate, use another documented equipment category for the pilot.
Expand with real artist layers and synthetic rigged renders after the data path is
verified. Combinatorial recomposition and recoloring do not create independent
equipment designs for generalization evaluation.

## First model and evaluation

Use a small conditional convolutional model as an engineering baseline. Inputs
include the target base frame, a canonical equipment reference, and its reference
base frame. Output is equipment RGB and alpha. This learns fitting rather than
character animation. It is not the planned final image generator and initially
has no temporal module.

Avoid a loss dominated by transparent pixels: supervise visible RGB, foreground-
balanced alpha, and premultiplied color/composition. Ignore arbitrary RGB values
under zero alpha. Report foreground alpha IoU and visible-color error, and compare
against empty-overlay and unchanged-reference baselines. Inspect previews because
scalar metrics alone miss design drift, misplaced attachments, and flicker.

Separate evaluations: tiny training-set overfitting, unseen equipment designs on
the known base, then unseen bodies/animations. Any temporal consistency measure
must account for movement and occlusion. The first pilot's matching frame-layout
restriction must be explicit in saved model metadata and inference validation.

## HPC and deployment constraints

Prepare/download data before submitting jobs. Use a single Iris GPU for the first
test. Check that the installed PyTorch/CUDA build actually supports V100 (compute
capability 7.0); use an appropriate cluster-provided stack or compatible build.
Do not assume the latest wheel supports that GPU. Begin in float32 for debugging;
profile supported mixed precision later. Keep module/environment selection explicit
because the user's available software stack and allocation are not known yet.

Record config, seed, code revision, dataset fingerprint, package versions, device,
timings, and memory with each run. Checkpoint weights and export equipment sheets
without requiring network access. Do not perform training on HPC login nodes.

Sources: [Iris hardware](https://hpc-docs.uni.lu/systems/iris/compute/),
[ULHPC Slurm](https://hpc-docs.uni.lu/slurm/),
[PyTorch installation](https://pytorch.org/get-started/locally/),
[Universal LPC](https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator).
