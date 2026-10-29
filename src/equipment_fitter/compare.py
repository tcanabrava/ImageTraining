"""Fixed three-seed architecture comparison; no checkpoint selection on validation."""
import argparse
from pathlib import Path
from .data import save_json
from .pilot import run_pilot
from .appearance import analyze


def compare(manifest, output, device):
    if output.exists():
        raise ValueError("Choose a new comparison directory")
    output.mkdir(parents=True)
    results = []
    for seed in (42, 43, 44):
        for architecture in ("unet", "reference_transfer"):
            name = f"{architecture}-{seed}"
            config = output / f"{name}.json"
            save_json(config, {"architecture": architecture, "steps": 200, "batch_size": 8,
                               "width": 16, "learning_rate": .001, "seed": seed,
                               "log_every": 25, "threads": 4})
            run = output / name
            run_pilot(manifest, config, run, device)
            report = analyze(run / "last.pt", manifest, output / f"{name}-appearance", device)
            results.append({"architecture": architecture, "seed": seed, "metrics": report["splits"]})
            save_json(output / "comparison.json", {"steps": 200, "results": results,
                      "scope": "Three fixed seeds, final checkpoint; same tiny split, no independent-reference test."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    compare(args.manifest, args.output, args.device)
