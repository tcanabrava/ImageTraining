"""Small supervised fitting baseline; no pretrained weights or motion generation."""

import torch
from torch import nn
from torch.nn import functional as F


def block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU())


class EquipmentUNet(nn.Module):
    def __init__(self, width: int = 16, output_channels: int = 4):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("Model width must be a positive multiple of four")
        self.enc1 = block(14, width)
        self.enc2 = block(width, width * 2)
        self.middle = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.output = nn.Conv2d(width, output_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        a = self.enc1(inputs)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.middle(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="nearest"), b), dim=1))
        e = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode="nearest"), a), dim=1))
        return self.output(e)


def fitting_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.sigmoid()
    alpha = target[:, 3:4]
    predicted_alpha = prediction[:, 3:4]
    dims = (1, 2, 3)
    foreground = alpha.sum(dims).clamp_min(1)
    background = (1 - alpha).sum(dims).clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(logits[:, 3:4], alpha, reduction="none")
    alpha_loss = 0.5 * ((bce * alpha).sum(dims) / foreground +
                        (bce * (1 - alpha)).sum(dims) / background)
    dice = 1 - (2 * (predicted_alpha * alpha).sum(dims) + 1) / (
        predicted_alpha.sum(dims) + alpha.sum(dims) + 1)
    visible_rgb = ((prediction[:, :3] - target[:, :3]).abs() * alpha).sum(dims) / (3 * foreground)
    premult_error = (prediction[:, :3] * predicted_alpha - target[:, :3] * alpha).abs().mean(1, keepdim=True)
    premult_loss = 0.5 * ((premult_error * alpha).sum(dims) / foreground +
                          (premult_error * (1 - alpha)).sum(dims) / background)
    return (alpha_loss + dice + visible_rgb + premult_loss).mean()


class ReferenceTransfer(nn.Module):
    """Experimental visible-reference color transfer with learned spatial attention.

    Geometry and correspondence see body pixels, coordinates and reference alpha,
    but not equipment RGB. RGB is a convex combination of visible reference colors.
    This deliberately restricts appearance to the aligned reference palette.
    """
    def __init__(self, width=16):
        super().__init__()
        self.geometry = EquipmentUNet(width, width + 1)
        self.keys = block(6, width)
        self.width = width

    def forward(self, inputs):
        geometry_inputs = inputs.clone()
        geometry_inputs[:, 8:11] = 0
        spatial = self.geometry(geometry_inputs)
        keys = self.keys(torch.cat((inputs[:, 4:8], inputs[:, 12:14]), dim=1))
        outputs = []
        for i in range(len(inputs)):
            mask = inputs[i, 11].flatten() > 0
            if not mask.any():
                raise ValueError("Equipment reference is empty")
            query = spatial[i, :self.width].flatten(1).T
            key = keys[i].flatten(1)[:, mask]
            weights = (query @ key / self.width ** .5).softmax(dim=-1)
            alpha = inputs[i, 11].flatten()[mask]
            colors = inputs[i, 8:11].flatten(1)[:, mask].T / alpha[:, None]
            rgb = (weights @ colors).T.reshape(3, *inputs.shape[-2:])
            outputs.append(torch.cat((torch.logit(rgb.clamp(1e-6, 1 - 1e-6)), spatial[i, self.width:]), dim=0))
        return torch.stack(outputs)


def build_model(config):
    architecture = config.get("architecture", "unet")
    if architecture == "unet":
        return EquipmentUNet(config["width"])
    if architecture == "reference_transfer":
        return ReferenceTransfer(config["width"])
    raise ValueError(f"Unknown architecture: {architecture}")
