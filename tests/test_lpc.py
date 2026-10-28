import io
import json

import pytest
from PIL import Image

from equipment_fitter.data import load_dataset
from equipment_fitter.lpc import import_lpc


def test_import_retains_alpha_coordinates_and_credits(tmp_path, monkeypatch):
    config = {"revision": "a" * 40, "description": "test", "category": "glove", "animation": "walk",
              "view": "south", "frame_size": [8, 8], "duration_ms": 100, "reference_frame": 0,
              "source_frames": [[8, 0], [0, 0]], "base_layers": ["spritesheets/base.png"],
              "equipment": [{"id": "glove", "design_id": "glove", "split": "train", "path": "spritesheets/glove.png"}]}
    image = Image.new("RGBA", (16, 8))
    image.putpixel((1, 1), (255, 0, 0, 255))
    image.putpixel((9, 1), (0, 255, 0, 255))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    credits = b"filename,notes,authors,licenses,urls\nbase.png,,Artist,CC0,https://example.org\nglove.png,,Artist,CC0,https://example.org\n"
    monkeypatch.setattr("equipment_fitter.lpc.fetch_asset", lambda rev, name, cache: credits if name == "CREDITS.csv" else buf.getvalue())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(config))
    manifest = import_lpc(selection, tmp_path / "out", tmp_path / "cache")
    assert load_dataset(manifest)[1]["target_frames"]["train"] == 1
    with Image.open(manifest.parent / "glove.png") as result:
        assert result.getpixel((1, 1)) == (0, 255, 0, 255)
        assert result.getpixel((9, 1)) == (255, 0, 0, 255)
    assert "Artist" in (manifest.parent / "CREDITS.csv").read_text()
    assert len(json.loads((manifest.parent / "provenance.json").read_text())["assets"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        import_lpc(selection, tmp_path / "out", tmp_path / "cache")
