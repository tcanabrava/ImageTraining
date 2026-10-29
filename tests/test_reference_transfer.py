import torch
from equipment_fitter.model import ReferenceTransfer, build_model, EquipmentUNet, fitting_loss


def test_transfer_obeys_color_change_and_preserves_alpha_with_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = ReferenceTransfer(4)
    inputs = torch.rand(2, 14, 9, 11)
    inputs[:, 11] = 0
    inputs[:, 11, 2:4, 3:5] = 1
    inputs[:, 8:11] *= inputs[:, 11:12]
    output = model(inputs)
    changed = inputs.clone()
    changed[:, 8:11] = inputs[:, 11:12] - inputs[:, 8:11]
    recolored = model(changed).sigmoid()
    assert torch.allclose(recolored[:, :3], 1 - output.sigmoid()[:, :3], atol=2e-6)
    assert torch.equal(recolored[:, 3], output.sigmoid()[:, 3])
    target = torch.rand_like(output)
    fitting_loss(output, target).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.keys.parameters())


def test_old_configs_keep_original_architecture():
    assert isinstance(build_model({'width': 4}), EquipmentUNet)
