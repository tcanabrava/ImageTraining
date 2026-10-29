"""Offline appearance diagnostic; palette projection is an ablation, not a trained model."""
import argparse
import hashlib
from pathlib import Path

import torch

from .data import asset_path, frame, load_dataset, rgba, save_json
from .model import build_model
from .tensors import conditioning, image_tensor
from .training import environment, select_device


def project_palette(prediction, reference):
    """Nearest visible reference RGB in sRGB space, preserving predicted alpha."""
    visible = reference[3] > 0
    if not visible.any():
        raise ValueError("Reference palette is empty")
    palette = torch.unique(reference[:3, visible].T, dim=0)
    rgb = prediction[:3].permute(1, 2, 0).reshape(-1, 3)
    # Chunk to bound working memory if future inputs have larger palettes.
    mapped = torch.cat([palette[torch.cdist(chunk, palette).argmin(1)]
                        for chunk in rgb.split(1024)])
    result = prediction.clone()
    result[:3] = mapped.reshape(*prediction.shape[1:], 3).permute(2, 0, 1)
    return result


def exported(image):
    result = image.clone()
    result[3] = (result[3] >= 0.5).to(result.dtype)
    result = (result.clamp(0, 1) * 255).round() / 255
    result[:3] *= result[3:4]
    return result


def metrics(predictions, targets):
    p, t = torch.stack(predictions), torch.stack(targets)
    a, b = p[:, 3:4] >= .5, t[:, 3:4] >= .5
    intersection, union = (a & b).sum().item(), (a | b).sum().item()
    foreground = b.sum().item()
    rgb = (p[:, :3] * p[:, 3:4] - t[:, :3] * t[:, 3:4]).abs()
    return {"frames": len(targets), "alpha_iou": intersection / union if union else 1.,
            "alpha_recall": intersection / foreground if foreground else 1.,
            "premultiplied_mae": rgb.mean().item(),
            "target_foreground_composite_mae": (rgb * b).sum().item() / max(3 * foreground, 1)}


@torch.no_grad()
def analyze(checkpoint_path, manifest, output, requested_device="cpu"):
    if output.exists():
        raise ValueError("Choose a new diagnostic output directory")
    torch.set_num_threads(4)
    device = select_device(requested_device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    document, dataset = load_dataset(manifest)
    if checkpoint.get("format_version") != 1 or checkpoint.get("input_contract") not in ("aligned-reference-rgba-v1", "canonical-footwear-rgba-v1"):
        raise ValueError("Unsupported checkpoint")
    if checkpoint["dataset_fingerprint"] != dataset["fingerprint"]:
        raise ValueError("This diagnostic requires the checkpoint's original benchmark dataset")
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    report = {"environment": environment(device), "dataset": dataset,
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "checkpoint_step": checkpoint["step"], "designs": {}, "splits": {},
              "input_contract": checkpoint["input_contract"],
              "scope": "Same animation and frame-derived reference. Palette oracle uses targets only to measure a representation bound; never used for predictions."}
    pooled = {}
    for sample in document["samples"]:
        size = document["frame_size"]
        base = rgba(asset_path(manifest.parent, sample["base"]))
        item = rgba(asset_path(manifest.parent, sample["equipment"]))
        ref_index = sample["reference_frame"]
        ref = image_tensor(frame(item, sample, ref_index, size)).to(device)
        ref_base = image_tensor(frame(base, sample, ref_index, size))
        variants = {name: [] for name in ("model", "reference_palette", "copy_reference", "palette_oracle")}
        targets, sensitivity, inversion_errors = [], [], []
        for i in range(len(sample["frames"])):
            if i == ref_index:
                continue
            target_base = image_tensor(frame(base, sample, i, size))
            if checkpoint["input_contract"] == "canonical-footwear-rgba-v1":
                from .canonical import canonical_conditioning
                inputs = canonical_conditioning(target_base, ref.cpu(), checkpoint["config"]["attachment"]).to(device)
            else:
                inputs = conditioning(target_base, ref_base, ref.cpu()).to(device)
            pred = model(inputs[None]).sigmoid()[0]
            altered = inputs.clone()
            # Invert only reference equipment RGB. Body, alpha, and layout stay fixed.
            altered[8:11] = altered[11:12] - altered[8:11]
            changed = model(altered[None]).sigmoid()[0]
            target = image_tensor(frame(item, sample, i, size)).to(device)
            mask = target[3:4]
            denom = (3 * mask.sum()).clamp_min(1)
            sensitivity.append(((changed[:3] - pred[:3]).abs() * mask).sum().item() / denom.item())
            inversion_errors.append(((changed[:3] - (1 - pred[:3])).abs() * mask).sum().item() / denom.item())
            targets.append(exported(target).cpu())
            for name, value in (("model", pred), ("reference_palette", project_palette(pred, ref)),
                                ("copy_reference", ref), ("palette_oracle", project_palette(target, ref))):
                variants[name].append(exported(value).cpu())
        result = {"split": sample["split"], "metrics": {k: metrics(v, targets) for k, v in variants.items()},
                  "inverted_reference_mean_rgb_change": sum(sensitivity) / len(sensitivity),
                  "inversion_equivariance_rgb_mae": sum(inversion_errors) / len(inversion_errors)}
        report["designs"][sample["id"]] = result
        pool = pooled.setdefault(sample["split"], {"targets": [], **{k: [] for k in variants}})
        pool["targets"].extend(targets)
        for name, values in variants.items():
            pool[name].extend(values)
    for split, values in pooled.items():
        report["splits"][split] = {k: metrics(v, values["targets"]) for k, v in values.items() if k != "targets"}
    output.mkdir(parents=True)
    save_json(output / "appearance.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    analyze(args.checkpoint, args.manifest, args.output, args.device)
    print(f"Appearance diagnostic complete: {args.output}/appearance.json")
