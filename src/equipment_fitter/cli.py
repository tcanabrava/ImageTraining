import argparse
import json
from pathlib import Path

from .data import load_dataset
from .fixture import create_fixture
from .preview import preview_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit equipment to existing spritesheets")
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture", help="Create geometric pipeline-test data")
    fixture.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Validate paired RGBA data")
    validate.add_argument("manifest", type=Path)
    preview = commands.add_parser("preview", help="Export base/equipment/composite contact sheets and GIFs")
    preview.add_argument("manifest", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--allow-empty", action="store_true", help="Permit empty model predictions for visual inspection")
    fetch = commands.add_parser("fetch-lpc", help="Download a pinned, explicitly selected LPC subset")
    fetch.add_argument("--selection", type=Path, required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--cache", type=Path, default=Path("data/cache/lpc"))
    train = commands.add_parser("train", help="Train the compact supervised fitting baseline")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--resume", type=Path)
    export = commands.add_parser("export-input", help="Extract a standalone inference input from a benchmark")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--sample", required=True)
    export.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("infer", help="Fit equipment from a base sheet and one aligned reference frame")
    for name in ("checkpoint", "base", "reference", "reference-base", "layout", "output"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    infer.add_argument("--threads", type=int, default=4)
    infer.add_argument("--alpha-mode", choices=["binary", "soft"], default="binary")
    pilot = commands.add_parser("pilot", help="Offline train, export predictions, and create previews")
    pilot.add_argument("--manifest", type=Path, required=True)
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            print(create_fixture(args.output))
        elif args.command == "validate":
            print(json.dumps(load_dataset(args.manifest)[1], indent=2))
        elif args.command == "preview":
            preview_dataset(args.manifest, args.output, allow_empty=args.allow_empty)
            print(args.output)
        elif args.command == "fetch-lpc":
            from .lpc import import_lpc
            print(import_lpc(args.selection, args.output, args.cache))
        elif args.command == "train":
            from .training import train
            train(args.manifest, args.config, args.output, args.device, args.resume)
        elif args.command == "export-input":
            from .inference import export_input
            export_input(args.manifest, args.sample, args.output)
            print(args.output)
        elif args.command == "infer":
            from .inference import infer
            report = infer(args.checkpoint, args.base, args.reference, args.reference_base,
                           args.layout, args.output, args.device, args.threads, args.alpha_mode)
            print(json.dumps(report, indent=2))
        elif args.command == "pilot":
            from .pilot import run_pilot
            run_pilot(args.manifest, args.config, args.output, args.device)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
