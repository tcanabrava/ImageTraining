"""Verify the actual module/venv stack and CUDA kernels before a training run."""

import json
import re
import sys


def check_version(name, actual, minimum, maximum):
    match = re.match(r"(\d+)\.(\d+)", actual)
    if not match or not minimum <= tuple(map(int, match.groups())) < maximum:
        raise RuntimeError(f"Unsupported {name} {actual}; expected >= {minimum} and < {maximum}")


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11+ is required")
    import numpy as np
    import PIL
    from PIL import Image
    import torch

    check_version("PyTorch", str(torch.__version__), (2, 2), (3, 0))
    check_version("NumPy", np.__version__, (1, 26), (3, 0))
    check_version("Pillow", PIL.__version__, (10, 0), (14, 0))
    from equipment_fitter.model import EquipmentUNet, fitting_loss
    from equipment_fitter.training import environment, select_device

    # Catch NumPy ABI mismatches as well as missing GPU/CUDA support.
    pixels = np.asarray(Image.new("RGBA", (16, 16)), dtype=np.float32).copy()
    torch.from_numpy(pixels).numpy()
    device = select_device("cuda")
    torch.set_num_threads(1)
    model = EquipmentUNet(4).to(device)
    inputs = torch.zeros(1, 14, 16, 16, device=device)
    target = torch.zeros(1, 4, 16, 16, device=device)
    loss = fitting_loss(model(inputs), target)
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("GPU forward/backward check produced nonfinite values")
    torch.cuda.synchronize(device)
    print(json.dumps({"preflight": "passed", **environment(device)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"HPC preflight failed: {error}", file=sys.stderr)
        print("Training was not started. Check the selected GPU PyTorch module and inherited NumPy/Pillow packages; see hpc/README.md.", file=sys.stderr)
        sys.exit(1)
