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
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            print(create_fixture(args.output))
        elif args.command == "validate":
            print(json.dumps(load_dataset(args.manifest)[1], indent=2))
        elif args.command == "preview":
            preview_dataset(args.manifest, args.output)
            print(args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
