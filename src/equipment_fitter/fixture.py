"""Procedural geometric data to exercise the pipeline, not an art benchmark."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import save_json


def create_fixture(output: Path) -> Path:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    size, count = 32, 4
    base = Image.new("RGBA", (size * count, size))
    hands = [(8, 19), (11, 16), (21, 16), (24, 20)]
    for i, (x, y) in enumerate(hands):
        tile = Image.new("RGBA", (size, size))
        draw = ImageDraw.Draw(tile)
        draw.line([(16, 13), (x, y)], fill=(210, 150, 110, 255), width=3)
        draw.rectangle((13, 11, 19, 26), fill=(55, 90, 140, 255))
        draw.ellipse((12, 3, 20, 11), fill=(210, 150, 110, 255))
        base.paste(tile, (i * size, 0))
    base.save(output / "base.png")
    samples = []
    for design, color in enumerate([(220, 50, 50), (40, 190, 100), (220, 180, 40)]):
        equipment = Image.new("RGBA", base.size)
        for i, (x, y) in enumerate(hands):
            tile = Image.new("RGBA", (size, size))
            draw = ImageDraw.Draw(tile)
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=(*color, 255))
            if design == 1:
                draw.rectangle((x - 1, y + 2, x + 1, y + 4), fill=(*color, 255))
            elif design == 2:
                draw.rectangle((x - 3, y - 3, x + 3, y - 2), fill=(*color, 255))
            # The torso occludes the glove in this synthetic example.
            pixels = np.array(tile)
            pixels[11:27, 13:20, 3] = 0
            equipment.paste(Image.fromarray(pixels), (i * size, 0))
        filename = f"glove-{design}.png"
        equipment.save(output / filename)
        samples.append({"id": f"glove-{design}", "design_id": f"fixture-shape-{design}",
                        "split": "val" if design == 2 else "train", "category": "glove",
                        "base": "base.png", "equipment": filename, "reference_frame": 0,
                        "frames": [{"xy": [i * size, 0], "duration_ms": 140} for i in range(count)]})
    manifest = output / "dataset.json"
    save_json(manifest, {"schema_version": 1, "frame_size": [size, size],
                         "description": "Geometric pipeline fixture; not evidence of sprite generalization.",
                         "samples": samples})
    return manifest
