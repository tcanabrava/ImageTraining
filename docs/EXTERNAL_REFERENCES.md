# Separately authored paired references

The pipeline now supports a separate reference image for canonical models. This is
an input/data capability, not evidence that a model handles independent artwork.
There is currently **no accepted real external-reference benchmark** in this project.

Add an optional `reference` object to a normal dataset sample:

```json
"reference": {
  "kind": "separately_authored",
  "image": "references/item.png",
  "provenance": "references/item.json"
}
```

The image must be nonempty RGBA at the dataset frame size. Provenance must contain
`creator`, `source` (source URL or full prompt/model details), `rights`,
`design_description`, `method` (`human_drawn` or `text_to_image`), `image_sha256`,
and a `pairing_review` with `status: accepted`, `reviewer`, and substantive `notes`.
Target inspection informing a text prompt must be disclosed; do not call that an
independently invented matching design. Rights records are declarations, not an
automated licensing determination.

Reference images and provenance bytes enter the dataset fingerprint. Validation
rejects missing/pending reviews, altered image hashes, incorrect sizes, empty
references, and exact visible target-frame duplicates even after translation.
It cannot establish authorship, detect every resize/edit/trace, or determine that
two drawings actually depict the same item. A review remains necessary.

Training reads the external image instead of cropping equipment targets. Only
`canonical_transfer` accepts it: no reference body is fabricated from the targets.
Export preserves original provenance and saves the supplied image, and canonical
pilot inference centers it without a reference-body file. Inference still receives
no target sheets. Prediction manifests omit the external-reference declarations
because those declarations describe supervision, not predicted assets.

For backwards-compatible evaluation, `reference_frame` remains an excluded target
index even with external art. It does not determine the external input image.
This sacrifices one target frame but keeps comparisons with old runs consistent;
a future all-target protocol must explicitly change that choice. The copy-reference
baseline simply copies the supplied image and does not assume it is registered.

## Preparing review dossiers

```sh
PYTHONPATH=src python -m equipment_fitter.authoring \
  --manifest data/reference-authoring-targets/dataset.json \
  --output data/reference-authoring-review
```

This copies targets with their source credits/provenance, exports target contact
sheets, and creates pending per-sample reference metadata plus a deliberately
untrainable draft manifest. Add actual separate artwork, review it, and then run:

```sh
PYTHONPATH=src python -m equipment_fitter validate \
  data/reference-authoring-review/draft-dataset.json
```

Review matching silhouette, construction, color placement, component count and
view—not just category. Prefer agreeing a design specification first, then having
references and animation layers authored against it. A text-only redraw of an
existing sprite is unlikely to reproduce all details. Do not invent matching
labels or use model predictions as ground-truth equipment layers.

## First authoring trial

`configs/lpc-reference-authoring.json` pins a credited lace-cuff target subset.
It is an unscored authoring trial, not a new frozen test. The built-in image_gen
tool generated a candidate without image inputs, using a text description informed
by inspection of the targets. The exact [prompt](REFERENCE_CANDIDATE_PROMPT.txt)
and [review](benchmarks/reference-authoring-trial.json) are versioned. The candidate
is saved locally at `data/reference-authoring-candidates/lace-cuffs-v1.png`.

**Rejected:** the candidate has broad ornate frills, whereas targets are tiny,
narrow bands. It also has a 1536×1024 canvas instead of the required 64×64. It is
RGBA with transparency, but is not an accepted same-design pair. No resizing or
acceptance was used to hide the mismatch. No training/HPC model scores were run on
it. The draft review dossier remains pending; all assets under data/ remain ignored
by Git. The geometric software-test fixtures are not real paired-reference data.

Next obtain a coherent design-first reference/layer pair (artist-authored, or a
reviewed reference followed by separately authored target layers), keep new family
splits isolated, and freeze a fresh evaluation set. The consumed slippers/tabi
set remains unavailable as untouched evidence for subsequent model changes.
