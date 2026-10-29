import torch
import pytest
from equipment_fitter.appearance import exported, metrics, project_palette


def test_palette_ignores_hidden_rgb_and_preserves_alpha():
    reference = torch.zeros(4, 2, 2)
    reference[:3] = .5  # Hidden gray must not enter the palette.
    reference[:, 0, 0] = torch.tensor([1., 0., 0., 1.])
    prediction = torch.full((4, 2, 2), .5)
    mapped = project_palette(prediction, reference)
    assert torch.equal(mapped[3], prediction[3])
    assert torch.all(mapped[0] == 1) and torch.all(mapped[1:3] == 0)
    with pytest.raises(ValueError, match="empty"):
        project_palette(prediction, torch.zeros_like(reference))


def test_export_metrics_penalize_missing_pixels_and_round_like_png():
    target = torch.ones(4, 2, 2)
    prediction = target.clone()
    prediction[3, 0, 0] = .49
    prediction[:3, 1, 1] = .501
    rendered = exported(prediction)
    assert torch.all(rendered[:, 0, 0] == 0)
    assert rendered[0, 1, 1] == 128 / 255
    result = metrics([rendered], [target])
    assert result['alpha_iou'] == .75
    assert result['alpha_recall'] == .75
    assert result['target_foreground_composite_mae'] > .25
