"""Offline train/export/preview entry point for the first HPC job."""

import json
from pathlib import Path

from .data import load_dataset, save_json
from .inference import export_input, infer
from .preview import preview_dataset
from .training import train


def run_pilot(manifest: Path, config: Path, output: Path, device: str) -> None:
    document, _ = load_dataset(manifest)
    train(manifest, config, output, device)
    settings = json.loads(config.read_text())
    threads = settings.get("threads", 4)
    prediction_manifest = {**document, "description": "Model predictions, including the reference frame; metrics exclude reference frames.", "samples": []}
    for sample in document["samples"]:
        inputs = output / "inputs" / sample["id"]
        predictions = output / "predictions" / sample["id"]
        export_input(manifest, sample["id"], inputs)
        if settings.get("architecture") == "canonical_transfer":
            from PIL import Image
            from .canonical import center_reference
            from .tensors import image_tensor
            with Image.open(inputs / "reference.png") as source:
                pixels = (center_reference(image_tensor(source)).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            Image.fromarray(pixels).save(inputs / "reference.png")
            (inputs / "reference-base.png").unlink()
            layout = json.loads((inputs / "layout.json").read_text())
            layout["source"] = "Canonical frame-derived item; no reference body; not independent artwork."
            save_json(inputs / "layout.json", layout)
        infer(output / "last.pt", inputs / "base.png", inputs / "reference.png",
              inputs / "reference-base.png", inputs / "layout.json", predictions, device, threads)
        prediction_manifest["samples"].append({**sample,
            "base": (inputs / "base.png").relative_to(output).as_posix(),
            "equipment": (predictions / "equipment.png").relative_to(output).as_posix()})
    path = output / "prediction-dataset.json"
    save_json(path, prediction_manifest)
    # An empty prediction is a valid failure to inspect, not invalid training data.
    preview_dataset(path, output / "previews", allow_empty=True)
    preview_dataset(manifest, output / "ground-truth-previews")
    if (manifest.parent / "CREDITS.csv").is_file():
        (output / "CREDITS.csv").write_bytes((manifest.parent / "CREDITS.csv").read_bytes())
    print(f"Pilot complete: {output / 'summary.json'}; compare previews/ with ground-truth-previews/", flush=True)
