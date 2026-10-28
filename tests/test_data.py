import json

import numpy as np
import pytest
from PIL import Image

from equipment_fitter.data import asset_path, frame, load_dataset
from equipment_fitter.fixture import create_fixture
from equipment_fitter.preview import preview_dataset


def rewrite(path, change):
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


def test_fixture_validation_and_content_fingerprint(tmp_path):
    path = create_fixture(tmp_path / "data")
    _, before = load_dataset(path)
    assert before["target_frames"] == {"train": 6, "val": 3, "test": 0}
    image = Image.open(path.parent / "glove-0.png")
    image.putpixel((0, 0), (1, 2, 3, 255))
    image.save(path.parent / "glove-0.png")
    assert load_dataset(path)[1]["fingerprint"] != before["fingerprint"]


def test_rejects_design_leakage(tmp_path):
    path = create_fixture(tmp_path / "data")
    rewrite(path, lambda d: d["samples"][2].update(design_id=d["samples"][0]["design_id"]))
    with pytest.raises(ValueError, match="leakage"):
        load_dataset(path)


@pytest.mark.parametrize("xy", [[120, 0], [0, 1], [-1, 0]])
def test_rejects_bad_frame_geometry(tmp_path, xy):
    path = create_fixture(tmp_path / "data")
    rewrite(path, lambda d: d["samples"][0]["frames"][1].update(xy=xy))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_rejects_traversal(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        asset_path(tmp_path, "../outside.png")


def test_preview_preserves_base_and_hidden_pixels(tmp_path):
    path = create_fixture(tmp_path / "data")
    before = (path.parent / "base.png").read_bytes()
    preview_dataset(path, tmp_path / "preview")
    assert (path.parent / "base.png").read_bytes() == before
    doc, _ = load_dataset(path)
    sample = doc["samples"][0]
    with Image.open(path.parent / sample["base"]) as b, Image.open(path.parent / sample["equipment"]) as e:
        merged = np.array(Image.alpha_composite(b, e))
        assert np.array_equal(merged[np.array(e)[..., 3] == 0], np.array(b)[np.array(e)[..., 3] == 0])
        assert frame(e, sample, 0, (32, 32)).size == (32, 32)
    assert (tmp_path / "preview/glove-0.gif").is_file()
