from pathlib import Path

from PIL import Image, ImageDraw

from .data import asset_path, frame, load_dataset, rgba


def checkerboard(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGBA", size, (220, 220, 220, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], 8):
        for x in range(0, size[0], 8):
            if (x // 8 + y // 8) % 2:
                draw.rectangle((x, y, x + 7, y + 7), fill=(175, 175, 175, 255))
    return image


def preview_dataset(manifest: Path, output: Path) -> None:
    document, _ = load_dataset(manifest)
    output.mkdir(parents=True, exist_ok=True)
    width, height = document["frame_size"]
    for sample in document["samples"]:
        base = rgba(asset_path(manifest.parent, sample["base"]))
        equipment = rgba(asset_path(manifest.parent, sample["equipment"]))
        sheet = Image.new("RGB", (width * len(sample["frames"]), height * 3))
        animation = []
        for index in range(len(sample["frames"])):
            b = frame(base, sample, index, (width, height))
            e = frame(equipment, sample, index, (width, height))
            combined = Image.alpha_composite(b, e)
            for row, tile in enumerate((b, e, combined)):
                tile = Image.alpha_composite(checkerboard(tile.size), tile).convert("RGB")
                sheet.paste(tile, (index * width, row * height))
            animation.append(Image.alpha_composite(checkerboard(combined.size), combined).convert("RGB"))
        sheet.resize((sheet.width * 3, sheet.height * 3), Image.Resampling.NEAREST).save(output / f"{sample['id']}.png")
        animation = [f.resize((width * 4, height * 4), Image.Resampling.NEAREST) for f in animation]
        animation[0].save(output / f"{sample['id']}.gif", save_all=True, append_images=animation[1:],
                          duration=[f["duration_ms"] for f in sample["frames"]], loop=0, disposal=2)
