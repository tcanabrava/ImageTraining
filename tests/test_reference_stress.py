import numpy as np
from PIL import Image
import pytest
from equipment_fitter.reference_stress import transform_reference, translate_visible


def test_registration_controls_preserve_visible_colors():
    reference = Image.new('RGBA', (64, 64))
    reference.paste((12, 34, 56, 255), (20, 40, 24, 44))
    body = Image.new('RGBA', (64, 64))
    body.paste((100, 120, 140, 255), (16, 8, 40, 56))
    original = reference.tobytes(), body.tobytes()
    for name in ('aligned', 'pair_shift', 'item_shift', 'centered_no_body'):
        r, b = transform_reference(reference, body, name)
        pixels = np.asarray(r)
        assert np.count_nonzero(pixels[..., 3]) == 16
        assert np.all(pixels[pixels[..., 3] > 0] == (12, 34, 56, 255))
        if name == 'centered_no_body':
            assert r.getbbox() == (30, 30, 34, 34)
            assert b.getbbox() is None
        if name in ('aligned', 'item_shift'):
            assert b.tobytes() == body.tobytes()
    assert original == (reference.tobytes(), body.tobytes())


def test_refuses_clipping_and_empty_references():
    image = Image.new('RGBA', (8, 8), (255, 0, 0, 255))
    with pytest.raises(ValueError, match='clip'):
        translate_visible(image, 1, 0)
    with pytest.raises(ValueError, match='empty'):
        transform_reference(Image.new('RGBA', (8, 8)), image, 'aligned')
