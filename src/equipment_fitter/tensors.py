from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import asset_path, frame, load_dataset, rgba, reference_pair


def image_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0


def conditioning(base: torch.Tensor, reference_base: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    def premultiply(image):
        return torch.cat((image[:3] * image[3:4], image[3:4]), dim=0)
    height, width = base.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
    return torch.cat((premultiply(base), premultiply(reference_base), premultiply(reference), x[None], y[None]))


class FrameDataset(Dataset):
    """In-memory tiny-pilot dataset. Reference frames never become target examples."""

    def __init__(self, manifest: Path, split: str, architecture="unet", attachment="none"):
        document, self.report = load_dataset(manifest)
        self.size = tuple(document["frame_size"])
        if min(self.size) < 8:
            raise ValueError("Training frame dimensions must be at least 8 pixels")
        self.items = []
        for sample in document["samples"]:
            if sample["split"] != split:
                continue
            base = rgba(asset_path(manifest.parent, sample["base"]))
            equipment = rgba(asset_path(manifest.parent, sample["equipment"]))
            ref_index = sample["reference_frame"]
            if "reference" in sample and architecture != "canonical_transfer":
                raise ValueError("Separately authored references require canonical_transfer")
            ref_image, body_image = reference_pair(manifest.parent, sample, self.size)
            ref_base, reference = image_tensor(body_image), image_tensor(ref_image)
            for i in range(len(sample["frames"])):
                if i == ref_index:
                    continue
                target_base = image_tensor(frame(base, sample, i, self.size))
                target = image_tensor(frame(equipment, sample, i, self.size))
                if architecture == "canonical_transfer":
                    from .canonical import canonical_conditioning
                    inputs = canonical_conditioning(target_base, reference, attachment)
                else:
                    inputs = conditioning(target_base, ref_base, reference)
                self.items.append((inputs, target, reference.clone()))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
