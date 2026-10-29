# Dataset contract v1

A dataset directory contains `dataset.json` and explicitly RGBA PNG spritesheets.
All image paths are relative to that directory. Downloaded attribution metadata
belongs beside the assets. Never commit datasets to Git.

```json
{
  "schema_version": 1,
  "frame_size": [64, 64],
  "samples": [{
    "id": "glove-a-walk",
    "design_id": "glove-a",
    "split": "train",
    "category": "glove",
    "base": "base.png",
    "equipment": "glove-a.png",
    "reference_frame": 0,
    "frames": [
      {"xy": [0, 0], "duration_ms": 100},
      {"xy": [64, 0], "duration_ms": 100}
    ]
  }]
}
```

`base` and `equipment` must have identical dimensions. `frames` lists nonoverlapping
frame rectangles in playback order; `frame_size` defines each rectangle's size.
Transparent equipment pixels are hidden when composited over the base. Optional
metadata can record animation name, view, anchors, license, and provenance.

The pilot conditions on the equipment frame at `reference_frame` and the matching
base frame. Other frames are training/evaluation targets. This intentionally tests
reference-to-other-pose fitting; it does not yet accept a separate generated item
illustration. All recolors/variants of one design must share `design_id` and split.
The validator catches inconsistent group assignment, not semantic duplicates; the
asset selection still needs manual review.

Preview PNG rows are base, equipment, and composite. GIFs are checkerboard-backed
inspection previews; RGBA PNG sheets are the actual assets.

Run without installing the package:

```sh
export PYTHONPATH="$PWD/src"
python -m equipment_fitter fixture --output data/fixture
python -m equipment_fitter validate data/fixture/dataset.json
python -m equipment_fitter preview data/fixture/dataset.json --output runs/fixture-preview
python -m pytest
```

## Real-asset pilots

Download on a network-enabled preparation machine, then transfer the prepared
directory to the HPC. Imports pin an upstream revision and retain per-asset credits,
source URLs, and hashes. Existing output directories are never overwritten.

```sh
python -m equipment_fitter fetch-lpc --selection configs/lpc-glove.json --output data/lpc-glove
python -m equipment_fitter preview data/lpc-glove/dataset.json --output runs/lpc-glove-preview
python -m equipment_fitter fetch-lpc --selection configs/lpc-footwear.json --output data/lpc-footwear
python -m equipment_fitter preview data/lpc-footwear/dataset.json --output runs/lpc-footwear-preview
```

The glove pilot has one design and **no validation split**: use it only to diagnose
whether the model can learn the training set. The footwear pilot has three training
designs and two held-out designs, on the same body and south-facing walk. These are
small engineering benchmarks; neither establishes arbitrary-equipment capability.
The eight walking frames use source columns 1–8, excluding the idle column.

Source artwork uses mixed licenses; retain the generated `CREDITS.csv` and review
the listed terms before distributing source or modified artwork. The importer does
not relicense assets or decide licensing questions about trained models. It preserves
source alpha and does not automatically repair occlusion. Visually review targets.

## Expanded four-view footwear benchmark

`configs/lpc-footwear-expanded.json` pins the same upstream revision and imports
north/west/south/east walks, with a separate aligned reference per view. It has
40 samples, 10 named designs grouped into 9 families: revised boots/shoes share
one family. Training has 168 non-reference frames, validation 56, reserved test 56.

Training adds sandals (thin strap detail) and revised shoes/boots to the previous
three designs. Plate boots and ghillies remain development validation, not fresh
test designs. Slippers and tabi socks are reserved for a later frozen-model test.
Rimmed boots are excluded from training because their credits share the kimono
source with tabi socks. Shared upstream artists and ancestry across the entire
collection remain a limitation; these are family splits, not artist splits.

```sh
PYTHONPATH=src python -m equipment_fitter fetch-lpc \
  --selection configs/lpc-footwear-expanded.json --output data/lpc-footwear-expanded
sbatch hpc/compare.sbatch data/lpc-footwear-expanded/development.json 1600
```

`dataset.json` includes all splits for data audit. `development.json` includes
only train/validation; `reserved-test.json` only test. The comparison runner rejects
manifests containing test samples. Do not use the full manifest for routine
training/diagnostics. Data inspection is allowed for integrity, but reserved-test
model scores must wait until the model/protocol are frozen. Preserve all nested
per-view selections, source hashes, and credit files when transferring data.

The planned expanded comparison uses the same three seeds and batch size, with
1,600 steps fixed before evaluation: eight times the 200-step pilot budget for
eight times its training-frame count. This approximately matches sampling exposure,
not elapsed time. Metrics across different datasets are not directly comparable.
All four views occur in training and validation: this is not unseen-view testing.
A reference is supplied for each view; this does not demonstrate cross-view synthesis.

## Separately authored references

Canonical models can now consume an explicit external reference image with reviewed
provenance. See [the external-reference contract and authoring workflow](EXTERNAL_REFERENCES.md).
No accepted real paired benchmark exists yet; the first generated candidate was rejected.
