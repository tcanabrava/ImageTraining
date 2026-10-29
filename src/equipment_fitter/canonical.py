"""Canonical reference and body-derived attachment inputs; no target equipment access."""
import torch
from .tensors import conditioning


def center_reference(reference):
    visible = (reference[3] > 0).nonzero()
    if not len(visible):
        raise ValueError('Equipment reference is empty')
    y0, x0 = visible.min(0).values.tolist()
    y1, x1 = (visible.max(0).values + 1).tolist()
    h, w = reference.shape[-2:]
    top, left = (h - (y1-y0)) // 2, (w - (x1-x0)) // 2
    result = torch.zeros_like(reference)
    result[:, top:top+y1-y0, left:left+x1-x0] = reference[:, y0:y1, x0:x1]
    return result


def attachment_mask(base, mode):
    h, w = base.shape[-2:]
    if mode == 'none':
        return torch.ones((1, h, w), dtype=base.dtype, device=base.device)
    if mode != 'feet':
        raise ValueError('Unknown attachment mode')
    visible = (base[3] > 0).nonzero()
    if not len(visible):
        raise ValueError('Base body is empty')
    y0, x0 = visible.min(0).values.tolist()
    y1, x1 = (visible.max(0).values + 1).tolist()
    # Broad lower third of the body bounds, padded by two pixels. This is a
    # footwear-specific heuristic, not a segmentation or target-equipment oracle.
    top = max(0, y1 - (y1-y0 + 2)//3 - 2)
    mask = torch.zeros((1, h, w), dtype=base.dtype, device=base.device)
    mask[:, top:min(h,y1+2), max(0,x0-2):min(w,x1+2)] = 1
    return mask


def canonical_conditioning(base, reference, mode):
    ref = center_reference(reference)
    mask = attachment_mask(base, mode)
    # The four old reference-body channels now explicitly carry the target ROI.
    inputs = conditioning(base, torch.zeros_like(base), ref)
    inputs[4:8] = mask.expand(4, -1, -1)
    return inputs
