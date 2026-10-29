import torch
from equipment_fitter.canonical import center_reference, attachment_mask, canonical_conditioning
from equipment_fitter.model import CanonicalTransfer


def test_translation_invariance_and_body_only_attachment():
    ref = torch.zeros(4, 32, 32)
    ref[:, 5:8, 7:9] = 1
    shifted = torch.zeros_like(ref)
    shifted[:, 12:15, 10:12] = 1
    assert torch.equal(center_reference(ref), center_reference(shifted))
    base = torch.zeros_like(ref)
    base[:, 2:30, 8:24] = 1
    assert torch.equal(canonical_conditioning(base, ref, 'feet'), canonical_conditioning(base, shifted, 'feet'))
    mask = attachment_mask(base, 'feet')
    assert not mask[:, :18].any() and mask[:, 25, 10].item() == 1
    torch.set_num_threads(1)
    model = CanonicalTransfer(4)
    prediction = model(canonical_conditioning(base, ref, 'feet')[None]).sigmoid()[0]
    assert (prediction[3][mask[0] == 0] < 1e-7).all()
