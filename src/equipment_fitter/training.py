from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import torch

from .data import save_json
from .model import build_model, fitting_loss
from .tensors import FrameDataset


@dataclass
class TrainConfig:
    architecture: str = "unet"
    steps: int = 200
    batch_size: int = 8
    width: int = 16
    learning_rate: float = 0.001
    seed: int = 42
    log_every: int = 25
    threads: int = 4

    def validate(self):
        if self.architecture not in ("unet", "reference_transfer"):
            raise ValueError("Unknown architecture")
        for key in ("steps", "batch_size", "width", "log_every", "threads"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.width % 4:
            raise ValueError("width must be a multiple of four")


def select_device(requested: str) -> torch.device:
    name = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if name == "auto":
        name = "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda, or auto")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; inspect the PyTorch build and GPU allocation")
    device = torch.device(name)
    if name == "cuda":
        # Exercise kernels: an import alone does not prove V100 support in the build.
        check = torch.ones((8, 8), device=device)
        (check @ check).sum().item()
    return device


def environment(device: torch.device) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    result = {"python": platform.python_version(), "python_executable": sys.executable,
              "loaded_modules": [name for name in os.environ.get("LOADEDMODULES", "").split(":") if name],
              "torch": str(torch.__version__),
              "numpy": np.__version__, "pillow": PIL.__version__, "platform": platform.platform(),
              "cuda_build": torch.version.cuda, "device": str(device), "git_revision": git("rev-parse", "HEAD"),
              "git_dirty": None if status is None else bool(status)}
    if device.type == "cuda":
        result.update(gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)),
                      supported_architectures=torch.cuda.get_arch_list())
    return result


def batch(dataset, indices, device):
    values = [dataset[int(i)] for i in indices]
    return tuple(torch.stack([v[k] for v in values]).to(device) for k in range(3))


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=8, baseline=None):
    if not len(dataset):
        return None
    model.eval()
    intersection = union = rgb_error = foreground = premult_error = pixels = 0.0
    for start in range(0, len(dataset), batch_size):
        inputs, target, reference = batch(dataset, range(start, min(start + batch_size, len(dataset))), device)
        if baseline == "empty":
            predicted = torch.zeros_like(target)
        elif baseline == "copy_reference":
            predicted = reference
        else:
            predicted = model(inputs).sigmoid()
        alpha = target[:, 3:4]
        a, b = predicted[:, 3:4] >= 0.5, alpha >= 0.5
        intersection += (a & b).sum().item()
        union += (a | b).sum().item()
        rgb_error += ((predicted[:, :3] - target[:, :3]).abs() * alpha).sum().item()
        foreground += 3 * alpha.sum().item()
        premult_error += (predicted[:, :3] * predicted[:, 3:4] - target[:, :3] * alpha).abs().sum().item()
        pixels += target[:, :3].numel()
    return {"alpha_iou": intersection / union if union else 1.0,
            "visible_rgb_mae": rgb_error / foreground if foreground else 0.0,
            "premultiplied_mae": premult_error / pixels, "frames": len(dataset)}


def train(manifest: Path, config_path: Path, output: Path, requested_device: str, resume: Path | None = None):
    config = TrainConfig(**json.loads(config_path.read_text()))
    config.validate()
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(requested_device)
    training = FrameDataset(manifest, "train")
    validation = FrameDataset(manifest, "val")
    if not len(training):
        raise ValueError("Dataset has no training targets")
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError("Run directory is not empty; choose a new output or explicitly resume")
    model = build_model(asdict(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_step = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint["dataset_fingerprint"] != training.report["fingerprint"]:
            raise ValueError("Resume checkpoint is incompatible with this dataset")
        if checkpoint["config"].get("architecture", "unet") != config.architecture:
            raise ValueError("Resume config mismatch: architecture")
        for key in ("width", "batch_size", "learning_rate", "seed"):
            if checkpoint["config"][key] != getattr(config, key):
                raise ValueError(f"Resume config mismatch: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        if start_step >= config.steps:
            raise ValueError("Resume requires steps greater than the checkpoint step")
        # A new continuation directory keeps both runs' provenance intact.
        if output.exists() and any(output.iterdir()):
            raise ValueError("Resume into a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    run = {"config": asdict(config), "environment": environment(device), "dataset": training.report,
           "manifest": str(manifest.resolve()), "frame_size": list(training.size),
           "parameters": sum(p.numel() for p in model.parameters()),
           "resume": str(resume.resolve()) if resume else None, "initial_step": start_step}
    save_json(output / "run.json", run)
    save_json(output / "dataset.json", json.loads(manifest.read_text()))
    baselines = {name: {split: evaluate(model, data, device, config.batch_size, name)
                       for split, data in (("train", training), ("val", validation))}
                 for name in ("empty", "copy_reference")}
    save_json(output / "baselines.json", baselines)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimization_seconds = 0.0
    for step in range(start_step + 1, config.steps + 1):
        model.train()
        tick = time.perf_counter()
        indices = torch.randint(len(training), (config.batch_size,), generator=torch.Generator().manual_seed(config.seed + step))
        inputs, target, _ = batch(training, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = fitting_loss(model(inputs), target)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_seconds += time.perf_counter() - tick
        if step == start_step + 1 or step % config.log_every == 0 or step == config.steps:
            record = {"step": step, "loss": loss.item(),
                      "train": evaluate(model, training, device, config.batch_size),
                      "val": evaluate(model, validation, device, config.batch_size),
                      "elapsed_seconds": time.perf_counter() - started}
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            checkpoint = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "step": step, "config": asdict(config), "frame_size": list(training.size),
                          "dataset_fingerprint": training.report["fingerprint"],
                          "input_contract": "aligned-reference-rgba-v1"}
            temporary = output / "last.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "last.pt")
    summary = {**record, "optimization_seconds": optimization_seconds,
               "training_examples_per_second": (config.steps - start_step) * config.batch_size / optimization_seconds,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
               "scope": "Framewise supervised pilot; no temporal model or arbitrary-illustration support."}
    save_json(output / "summary.json", summary)
    return summary
