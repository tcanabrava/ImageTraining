import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .data import asset_path, frame, load_dataset, rgba, save_json, reference_pair
from .model import build_model
from .tensors import conditioning, image_tensor
from .training import environment, select_device


def export_input(manifest: Path, sample_id: str, output: Path) -> None:
    document, _ = load_dataset(manifest)
    sample = next((s for s in document["samples"] if s["id"] == sample_id), None)
    if sample is None:
        raise ValueError(f"Unknown sample id: {sample_id}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Input export directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    base = rgba(asset_path(manifest.parent, sample["base"]))
    equipment = rgba(asset_path(manifest.parent, sample["equipment"]))
    base.save(output / "base.png")
    size = document["frame_size"]
    ref, ref_body = reference_pair(manifest.parent, sample, size)
    if "reference" not in sample:
        ref_body.save(output / "reference-base.png")
    else:
        (output / "reference-provenance.json").write_bytes(asset_path(manifest.parent, sample["reference"]["provenance"]).read_bytes())
    ref.save(output / "reference.png")
    save_json(output / "layout.json", {"frame_size": size, "frames": sample["frames"],
              "reference_frame": sample["reference_frame"], "sample_id": sample_id,
              "source": "Separately authored reference with declared provenance." if "reference" in sample else "Aligned frame reference; not an independent item illustration."})
    if (manifest.parent / "CREDITS.csv").is_file():
        (output / "CREDITS.csv").write_bytes((manifest.parent / "CREDITS.csv").read_bytes())


@torch.no_grad()
def infer(checkpoint_path: Path, base_path: Path, reference_path: Path, reference_base_path: Path | None,
          layout_path: Path, output: Path, requested_device: str = "auto", threads: int = 4,
          alpha_mode: str = "binary") -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Inference output directory must be empty")
    if threads < 1:
        raise ValueError("threads must be positive")
    if alpha_mode not in ("binary", "soft"):
        raise ValueError("alpha_mode must be binary or soft")
    torch.set_num_threads(threads)
    device = select_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("input_contract") not in ("aligned-reference-rgba-v1", "canonical-footwear-rgba-v1"):
        raise ValueError("Unsupported checkpoint")
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    canonical = checkpoint["input_contract"] == "canonical-footwear-rgba-v1"
    base, reference = map(rgba, (base_path, reference_path))
    if not canonical and reference_base_path is None:
        raise ValueError("Aligned checkpoints require a reference body")
    reference_base = Image.new("RGBA", reference.size) if canonical else rgba(reference_base_path)
    layout = json.loads(layout_path.read_text())
    size = tuple(layout["frame_size"])
    if list(size) != checkpoint["frame_size"] or reference.size != size or reference_base.size != size:
        raise ValueError("Frame/reference size differs from the trained input contract")
    if reference.getchannel("A").getbbox() is None:
        raise ValueError("Equipment reference is empty")
    if not layout.get("frames"):
        raise ValueError("No frames in layout")
    rectangles = []
    for f in layout["frames"]:
        x, y = f["xy"]
        if any(type(v) is not int for v in (x, y)) or min(x, y) < 0 or x + size[0] > base.width or y + size[1] > base.height:
            raise ValueError("Frame outside base canvas")
        if type(f.get("duration_ms")) is not int or f["duration_ms"] <= 0:
            raise ValueError("Invalid frame duration")
        if any(x < r[2] and x + size[0] > r[0] and y < r[3] and y + size[1] > r[1] for r in rectangles):
            raise ValueError("Overlapping frame rectangles")
        rectangles.append((x, y, x + size[0], y + size[1]))
    ref, ref_base = image_tensor(reference), image_tensor(reference_base)
    sheet = Image.new("RGBA", base.size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for i, f in enumerate(layout["frames"]):
        b = image_tensor(frame(base, layout, i, size))
        if canonical:
            from .canonical import canonical_conditioning
            inputs = canonical_conditioning(b, ref, checkpoint["config"]["attachment"])
        else:
            inputs = conditioning(b, ref_base, ref)
        prediction = model(inputs.unsqueeze(0).to(device)).sigmoid()[0]
        if alpha_mode == "binary":
            prediction[3] = (prediction[3] >= 0.5).to(prediction.dtype)
        pixels = (prediction.cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        # Binary alpha suits the pixel-art pilot; soft alpha remains an explicit option.
        pixels[pixels[..., 3] == 0, :3] = 0
        sheet.paste(Image.fromarray(pixels), tuple(f["xy"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / "equipment.png")
    Image.alpha_composite(base, sheet).save(output / "composite.png")
    save_json(output / "layout.json", layout)
    report = {"environment": environment(device), "checkpoint": str(checkpoint_path.resolve()),
              "checkpoint_step": checkpoint["step"], "frames": len(layout["frames"]),
              "generation_seconds": elapsed, "includes_model_loading": False, "alpha_mode": alpha_mode}
    save_json(output / "inference.json", report)
    return report
