import json

import numpy as np
import pytest
from PIL import Image
import torch

from equipment_fitter.fixture import create_fixture
from equipment_fitter.inference import export_input, infer
from equipment_fitter.model import EquipmentUNet, fitting_loss
from equipment_fitter.tensors import FrameDataset
from equipment_fitter.training import train


def test_hidden_rgb_does_not_affect_loss():
    logits = torch.zeros(1, 4, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 2:4, 2:4] = 1
    changed = target.clone()
    changed[:, :3] = torch.where(target[:, 3:4] == 0, torch.rand_like(changed[:, :3]), changed[:, :3])
    assert torch.equal(fitting_loss(logits, target), fitting_loss(logits, changed))
    loss = fitting_loss(logits, target)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_empty_target_and_odd_dimensions_have_finite_gradients():
    torch.set_num_threads(1)
    model = EquipmentUNet(4)
    logits = model(torch.zeros(1, 14, 17, 19))
    assert logits.shape == (1, 4, 17, 19)
    fitting_loss(logits, torch.zeros_like(logits)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("architecture", ["unet", "reference_transfer"])
def test_training_resume_and_inference_without_target_sheet(tmp_path, architecture):
    manifest = create_fixture(tmp_path / "data")
    config = tmp_path / "config.json"
    values = {"architecture": architecture, "steps": 2, "batch_size": 2, "width": 4, "learning_rate": 0.001,
              "seed": 7, "log_every": 1, "threads": 1}
    config.write_text(json.dumps(values))
    result = train(manifest, config, tmp_path / "run", "cpu")
    assert result["train"]["frames"] == 6
    assert result["val"]["frames"] == 3
    dataset = FrameDataset(manifest, "train")
    assert len(dataset) == 6  # Two designs, three targets each; reference frames excluded.
    values["steps"] = 3
    config.write_text(json.dumps(values))
    train(manifest, config, tmp_path / "resumed", "cpu", tmp_path / "run/last.pt")
    train(manifest, config, tmp_path / "uninterrupted", "cpu")
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    uninterrupted = torch.load(tmp_path / "uninterrupted/last.pt", weights_only=True)
    for key in resumed["model"]:
        assert torch.equal(resumed["model"][key], uninterrupted["model"][key])
    inputs = tmp_path / "inputs"
    export_input(manifest, "glove-2", inputs)
    original = (inputs / "base.png").read_bytes()
    # Delete ground-truth equipment: inference must only require the exported reference.
    (manifest.parent / "glove-2.png").unlink()
    infer(tmp_path / "run/last.pt", inputs / "base.png", inputs / "reference.png",
          inputs / "reference-base.png", inputs / "layout.json", tmp_path / "prediction", "cpu", 1)
    assert (inputs / "base.png").read_bytes() == original
    with Image.open(tmp_path / "prediction/equipment.png") as p, Image.open(inputs / "base.png") as b:
        assert p.mode == "RGBA" and p.size == b.size
        expected = np.array(Image.alpha_composite(b, p))
    with Image.open(tmp_path / "prediction/composite.png") as c:
        assert np.array_equal(expected, np.array(c))
    layout = json.loads((inputs / "layout.json").read_text())
    layout["frames"][1]["xy"] = layout["frames"][0]["xy"]
    (inputs / "layout.json").write_text(json.dumps(layout))
    with pytest.raises(ValueError, match="Overlapping"):
        infer(tmp_path / "run/last.pt", inputs / "base.png", inputs / "reference.png",
              inputs / "reference-base.png", inputs / "layout.json", tmp_path / "invalid", "cpu", 1)
