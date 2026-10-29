"""Canonical reference ablation: with and without a body-derived footwear ROI."""
import argparse
from pathlib import Path
from .data import load_dataset, save_json
from .pilot import run_pilot
from .appearance import analyze


def compare(manifest, output, device, steps=200):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    document, _ = load_dataset(manifest)
    if any(s["split"] == "test" for s in document["samples"]):
        raise ValueError("Use a development manifest without reserved test samples")
    if output.exists():
        raise ValueError("Choose a new comparison directory")
    output.mkdir(parents=True)
    results = []
    for seed in (42, 43, 44):
        for attachment in ("none", "feet"):
            architecture = "canonical_transfer"
            name = f"canonical-{attachment}-{seed}"
            config = output / f"{name}.json"
            save_json(config, {"architecture": architecture, "attachment": attachment, "steps": steps, "batch_size": 8,
                               "width": 16, "learning_rate": .001, "seed": seed,
                               "log_every": 25, "threads": 4})
            run = output / name
            run_pilot(manifest, config, run, device)
            report = analyze(run / "last.pt", manifest, output / f"{name}-appearance", device)
            results.append({"architecture": architecture, "attachment": attachment, "seed": seed, "metrics": report["splits"]})
            save_json(output / "comparison.json", {"steps": steps, "results": results,
                      "scope": "Three fixed seeds, final checkpoint; same tiny split, no independent-reference test."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=1600)
    args = parser.parse_args()
    compare(args.manifest, args.output, args.device, args.steps)
