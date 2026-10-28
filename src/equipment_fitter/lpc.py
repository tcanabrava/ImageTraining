"""Import an explicitly selected, pinned subset of Universal LPC."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

from .data import asset_path, load_dataset, save_json

REPOSITORY = "LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator"


def fetch_asset(revision: str, name: str, cache: Path) -> bytes:
    target = asset_path(cache / revision, name)
    if target.exists():
        return target.read_bytes()
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "equipment-fitter/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        temp.write(data)
        temp_path = Path(temp.name)
    temp_path.replace(target)
    return data


def import_lpc(selection: Path, output: Path, cache: Path) -> Path:
    config = json.loads(selection.read_text())
    revision = config["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("LPC revision must be a full commit SHA")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lpc-", dir=output.parent) as directory:
        staging = Path(directory) / "dataset"
        staging.mkdir()
        credits_data = fetch_asset(revision, "CREDITS.csv", cache)
        credits = {row["filename"]: row for row in csv.DictReader(io.StringIO(credits_data.decode("utf-8-sig")))}
        used_credits, provenance = [], []
        paths = list(dict.fromkeys(config["base_layers"] + [x["path"] for x in config["equipment"]]))
        images = {}
        for name in paths:
            if not name.startswith("spritesheets/"):
                raise ValueError(f"Expected a spritesheets/ asset: {name}")
            asset_path(staging, name)  # Reject traversal before downloading.
            credit = credits.get(name.removeprefix("spritesheets/"))
            if not credit or not credit.get("licenses", "").strip() or not credit.get("authors", "").strip():
                raise ValueError(f"Missing attribution/license metadata for {name}")
            used_credits.append(credit)
            data = fetch_asset(revision, name, cache)
            with Image.open(io.BytesIO(data)) as image:
                images[name] = image.convert("RGBA")
            provenance.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                               "bytes": len(data), "url": f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{name}"})
        base = images[config["base_layers"][0]].copy()
        for name in config["base_layers"][1:]:
            if images[name].size != base.size:
                raise ValueError("Base layers have mismatched dimensions")
            base = Image.alpha_composite(base, images[name])
        size = config["frame_size"]
        coordinates = config["source_frames"]

        def pack(image: Image.Image) -> Image.Image:
            if image.size != base.size:
                raise ValueError("Source base/equipment canvas mismatch")
            result = Image.new("RGBA", (size[0] * len(coordinates), size[1]))
            for index, (x, y) in enumerate(coordinates):
                if x < 0 or y < 0 or x + size[0] > image.width or y + size[1] > image.height:
                    raise ValueError("Source frame outside canvas")
                result.paste(image.crop((x, y, x + size[0], y + size[1])), (index * size[0], 0))
            return result

        pack(base).save(staging / "base.png")
        samples = []
        for item in config["equipment"]:
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", item["id"]):
                raise ValueError("Invalid equipment id")
            filename = item["id"] + ".png"
            pack(images[item["path"]]).save(staging / filename)
            samples.append({"id": item["id"], "design_id": item["design_id"], "split": item["split"],
                            "category": config["category"], "base": "base.png", "equipment": filename,
                            "reference_frame": config["reference_frame"],
                            "frames": [{"xy": [i * size[0], 0], "duration_ms": config["duration_ms"]}
                                       for i in range(len(coordinates))]})
        save_json(staging / "dataset.json", {"schema_version": 1, "frame_size": size,
                  "description": config["description"], "animation": config["animation"],
                  "view": config["view"], "source_revision": revision, "samples": samples})
        save_json(staging / "selection.json", config)
        save_json(staging / "provenance.json", {"repository": REPOSITORY, "revision": revision,
                  "credits_sha256": hashlib.sha256(credits_data).hexdigest(), "assets": provenance})
        with (staging / "CREDITS.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "notes", "authors", "licenses", "urls"])
            writer.writeheader()
            writer.writerows(used_credits)
        load_dataset(staging / "dataset.json")
        staging.rename(output)
    return output / "dataset.json"
