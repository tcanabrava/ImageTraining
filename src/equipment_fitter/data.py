"""Portable dataset contract. No PyTorch or network access is needed here."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image


def asset_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("Asset paths must be relative strings")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset escapes dataset directory: {value}")
    return path


def rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise ValueError(f"Expected RGBA image: {path} (got {image.mode})")
        return image.copy()


def frame(image: Image.Image, sample: dict, index: int, size: tuple[int, int]) -> Image.Image:
    x, y = sample["frames"][index]["xy"]
    width, height = size
    return image.crop((x, y, x + width, y + height))


def load_dataset(path: str | Path) -> tuple[dict, dict]:
    """Validate manifests, files, frame geometry, and group-level split isolation."""
    path = Path(path)
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Expected dataset schema_version 1")
    size = document.get("frame_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("frame_size must contain two positive integers")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset must contain samples")
    groups, identifiers, files = {}, set(), set()
    images = {}
    counts = {split: 0 for split in ("train", "val", "test")}
    for sample in samples:
        identifier = sample.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier) or identifier in identifiers:
            raise ValueError(f"Invalid or duplicate sample id: {identifier}")
        identifiers.add(identifier)
        split = sample.get("split")
        group = sample.get("design_id")
        if split not in counts or not isinstance(group, str) or not group:
            raise ValueError(f"{identifier}: split and design_id are required")
        if group in groups and groups[group] != split:
            raise ValueError(f"Design leakage across splits: {group}")
        groups[group] = split
        if not isinstance(sample.get("category"), str) or not sample["category"]:
            raise ValueError(f"{identifier}: category is required")
        for key in ("base", "equipment"):
            target = asset_path(path.parent, sample[key])
            files.add(target)
            if target not in images:
                images[target] = rgba(target)
        base = images[asset_path(path.parent, sample["base"])]
        equipment = images[asset_path(path.parent, sample["equipment"])]
        if base.size != equipment.size:
            raise ValueError(f"{identifier}: base/equipment canvas mismatch")
        frames = sample.get("frames", [])
        reference = sample.get("reference_frame")
        if len(frames) < 2 or type(reference) is not int or not 0 <= reference < len(frames):
            raise ValueError(f"{identifier}: at least two frames and a valid reference_frame required")
        rectangles = []
        for f in frames:
            xy = f.get("xy")
            if not isinstance(xy, list) or len(xy) != 2 or any(type(v) is not int or v < 0 for v in xy):
                raise ValueError(f"{identifier}: invalid frame coordinates")
            x, y = xy
            w, h = size
            if x + w > base.width or y + h > base.height:
                raise ValueError(f"{identifier}: frame outside canvas")
            if type(f.get("duration_ms")) is not int or f["duration_ms"] <= 0:
                raise ValueError(f"{identifier}: invalid frame duration")
            rect = (x, y, x + w, y + h)
            if any(x < b[2] and x + w > b[0] and y < b[3] and y + h > b[1] for b in rectangles):
                raise ValueError(f"{identifier}: overlapping frames")
            rectangles.append(rect)
        if frame(equipment, sample, reference, size).getchannel("A").getbbox() is None:
            raise ValueError(f"{identifier}: empty equipment reference")
        if not any(frame(equipment, sample, i, size).getchannel("A").getbbox()
                   for i in range(len(frames)) if i != reference):
            raise ValueError(f"{identifier}: all non-reference targets are empty")
        counts[split] += len(frames) - 1
    digest = hashlib.sha256()
    digest.update(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    for file in sorted(files):
        digest.update(file.relative_to(path.parent.resolve()).as_posix().encode())
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return document, {"samples": len(samples), "designs": len(groups), "target_frames": counts,
                      "fingerprint": digest.hexdigest()}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
