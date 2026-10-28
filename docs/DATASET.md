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
