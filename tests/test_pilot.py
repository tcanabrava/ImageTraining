import json

from PIL import Image

from equipment_fitter.fixture import create_fixture
from equipment_fitter.pilot import run_pilot
from equipment_fitter.preview import preview_dataset


def test_offline_pilot_produces_reviewable_artifacts(tmp_path):
    manifest = create_fixture(tmp_path / "data")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"steps": 1, "batch_size": 1, "width": 4, "threads": 1}))
    output = tmp_path / "run"
    run_pilot(manifest, config, output, "cpu")
    assert (output / "summary.json").is_file()
    assert (output / "predictions/glove-2/equipment.png").is_file()
    assert (output / "previews/glove-2.gif").is_file()
    assert (output / "ground-truth-previews/glove-2.png").is_file()
    prediction_manifest = json.loads((output / "prediction-dataset.json").read_text())
    for sample in prediction_manifest["samples"]:
        p = output / sample["equipment"]
        with Image.open(p) as image:
            blank = Image.new("RGBA", image.size)
        blank.save(p)
    # Empty predictions are inspectable failures, not a reason to abort the report.
    preview_dataset(output / "prediction-dataset.json", output / "empty-previews", allow_empty=True)
    assert (output / "empty-previews/glove-0.gif").is_file()
