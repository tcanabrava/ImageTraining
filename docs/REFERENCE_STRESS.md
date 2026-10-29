# Reference registration stress protocol

This is a development diagnostic, not a benchmark of independently drawn artwork.
It reuses the six hash-frozen expanded-footwear checkpoints and only the plate
boots/ghillies development validation samples. It does not evaluate the consumed
slippers/tabi test families. No training, palette correction, or threshold changes.

Four conditions are fixed before scores are observed:

1. `aligned`: original equipment reference and corresponding body frame (control).
2. `pair_shift`: shift both reference images eight pixels right. Their mutual
   registration is preserved, but absolute reference coordinates change.
3. `item_shift`: shift only equipment eight pixels right; body stays unchanged.
   This deliberately violates registration while preserving equipment colors.
4. `centered_no_body`: center the equipment's alpha bounding box in the same-size
   canvas and provide a transparent reference-body placeholder. This approximates
   a standalone presentation, but remains derived from the same animation frame.

Transforms reject clipping and preserve visible source pixels without rescaling.
Each reference pair is saved separately with source hash and provenance declaring
`independently_drawn: false`. Target poses, pixels, frame order, and timing are
unchanged; non-reference targets are never used to build reference inputs.
The last two conditions are outside the models' trained aligned-reference contract.
Their purpose is to measure that gap, not imply supported independent-art inference.

Score final model outputs using the existing PNG-equivalent binary-alpha metrics.
Report every seed, architecture, and condition; compare means and paired changes
against `aligned`. Inspect seed 42 examples, without selecting a best seed. The
aligned control should match the previous development scores within numerical
precision. Run `sbatch hpc/reference-stress.sbatch`; artifacts go to
`runs/reference-stress-JOBID/`.

A successful result would only indicate robustness to these transformations.
Actual independent-reference evaluation still needs separately authored designs
with paired fitted targets, reference provenance, new held-out families, and a
reference contract that does not require an aligned body image. Broader equipment
and behind-torso occlusion remain separate outstanding tasks.
