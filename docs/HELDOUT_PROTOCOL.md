# Frozen footwear test v1

This protocol is committed before scoring the reserved designs. The machine-readable
freeze is `configs/heldout-footwear-v1.json`: exact dataset fingerprints and SHA-256
hashes of all six final checkpoints from Iris job 6257841 (1,600 updates, architectures
U-Net/reference transfer, seeds 42/43/44). No retraining, seed selection, validation
checkpoint selection, palette correction, augmentation, or threshold tuning.

Test: slippers and tabi socks, four walk directions, one aligned reference per
design/view, seven scored non-reference frames per sample (56 targets). Body,
animation, and views are shared with development; only these design families were
reserved. Data integrity previews were inspected before the freeze, but no model
scores on these designs were computed. Shared artists/asset ancestry remain a limit.

Primary metric: target-foreground composite MAE on actual exported PNGs (binary
alpha threshold 0.5, rounded 8-bit RGB). Report every seed and mean across seeds,
with pooled alpha IoU, recall, and whole-frame premultiplied MAE as secondary
metrics. Include copy-reference and empty-overlay baselines and per-design/view
scores. Inspect seed 42 previews for both architectures, without selecting the best
seed. Do not interpret temporal stability from contact sheets or count frames/seeds
as independent unseen designs. A win on two families is not broad generalization.

Run `sbatch hpc/heldout.sbatch` from the Iris project root. The evaluator checks all
hashes, architecture/seed/step, training fingerprints, frame sizes, and disjoint
family splits before running inference. Outputs and protocol copy are retained in
`runs/heldout-JOBID/`. Existing inference receives only exported base/reference
inputs; targets enter scoring separately. No optimizer or training function runs.

After evaluation, these designs are no longer an untouched test for future model
changes. Any changes informed by the result require a new independent evaluation
set. This test does not authorize or validate production deployment.
