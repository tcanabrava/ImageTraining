# Canonical footwear reference experiment

This development ablation trains a new input contract (`canonical-footwear-rgba-v1`)
from scratch. A same-size RGBA item image is centered by its visible bounding box
without rescaling. The reference-body input is removed, not inferred. The target
body, item, target coordinates, and an explicit attachment mask condition placement.
Reference attention keys use canonical item alpha and coordinates; colors still
come directly from reference pixels. Body pixels and motion are never generated.

The item is still derived from an animation frame. Paired feet and their source
pose remain embedded in the reference; this is not independently authored artwork,
a single isolated shoe, arbitrary scale support, or arbitrary-equipment capability.
Translation invariance is built into centering and is not a learned achievement.

Compare two variants at seeds 42/43/44, 1,600 updates, batch 8, same expanded
training/validation data as job 6257841:

- `none`: full-canvas attachment mask; no spatial restriction.
- `feet`: broad rectangle covering the bottom third of the target body's visible
  bounds, padded by two pixels. It is derived only from the target base alpha,
  never target equipment. The model receives this region and forces alpha outside
  it near zero (below PNG export threshold). This is an explicit footwear-specific
  heuristic, not supplied artist anchors or learned pose estimation.

Keep old aligned-reference results as context, not a one-factor architectural
ablation: removing reference-body inputs, canonicalizing items, and changing keys
are combined. Comparing `none` versus `feet` isolates the ROI input/output constraint
as a combined mechanism. Masks can forbid legitimate equipment; report target-pixel
coverage as an audit only, without modifying the mask based on validation targets.

Primary scores: PNG-equivalent foreground composite MAE, IoU, recall, and whole-frame
error on all development validation frames. Report all seeds and compare to the
frozen aligned transfer baseline. No consumed test-set scoring or checkpoint/seed
selection. Inspect seed 42 without selecting a best seed. Use
`sbatch hpc/canonical.sbatch`; preserve configs, checkpoints, metrics and previews.

Canonical checkpoints accept `infer` without `--reference-base`. Existing aligned
checkpoints still require it. Resume checks architecture and attachment settings.
This prototype does not imply that arbitrary standalone artwork is supported.


Completed experiment: Iris job `6258458`; see [results](RESULTS.md). To run a
canonical checkpoint without a reference body (paths shown for the recorded run):

```sh
PYTHONPATH=src python -m equipment_fitter infer \
  --checkpoint runs/canonical-6258458/canonical-none-42/last.pt \
  --base runs/canonical-6258458/canonical-none-42/inputs/plate-boots-south/base.png \
  --reference runs/canonical-6258458/canonical-none-42/inputs/plate-boots-south/reference.png \
  --layout runs/canonical-6258458/canonical-none-42/inputs/plate-boots-south/layout.json \
  --output runs/canonical-example --device cpu
```

Use a new output directory. The reference must still be RGBA at the trained frame
size; automatic centering does not resize or generate missing equipment parts.
