"""Inspect a copied checkpoint without running an optimizer."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy.spatial import cKDTree
from gaussians import read_ply
from model import PetModel
from train import BoundGaussians, FreeGaussians, render, ssim
from render_alignment import render_structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bite-source", required=True)
    parser.add_argument("--bite-fit", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ply", help="Inspect an appearance PLY with the completed checkpoint's fixed mesh")
    args = parser.parse_args()
    directory = Path(args.training)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if not all(torch.isfinite(value).all() for value in saved["pet"].values()) or not all(torch.isfinite(value).all() for value in saved["gaussians"].values()):
        raise RuntimeError("Checkpoint contains non-finite model or Gaussian values")
    pet = PetModel(args.bite_source, args.bite_fit)
    pet.load_state_dict(saved["pet"])
    configuration = json.loads((directory / "cameras.json").read_text())
    selected = json.loads((directory / "target-montage.json").read_text())
    resolution = configuration["resolution"]
    if saved["stage"] == "bound":
        bound = BoundGaussians(pet, read_ply(directory / "proxy_normalized.ply"))
        bound.load_state_dict(saved["gaussians"])
        with torch.no_grad():
            parameters = bound.parameters_at(pet())
    else:
        parameters = FreeGaussians(saved["gaussians"])
    original_parameters = parameters
    if args.ply:
        loaded = read_ply(args.ply)
        if not all(np.isfinite(value).all() for value in loaded.values()):
            raise RuntimeError("Appearance PLY contains non-finite values")
        parameters = {name: torch.as_tensor(value, dtype=torch.float32, device="cuda") for name, value in loaded.items()}
    appearance_label = "refined appearance" if args.ply else f"{saved['stage']} {saved['step']}"
    montage = Image.new("RGB", (resolution * 3, (resolution + 24) * 4), "white")
    draw = ImageDraw.Draw(montage)
    background = np.array([232, 234, 223]) / 255
    viewer_montage = Image.new("RGB", (resolution * 2, (resolution + 24) * 4), tuple((background * 255).astype(int)))
    viewer_draw = ImageDraw.Draw(viewer_montage)
    alpha_metrics = {}
    with torch.no_grad():
        vertices = pet()
        for row, (name, view_id) in enumerate(selected.items()):
            view = configuration["views"][view_id]
            rgba = np.asarray(Image.open(directory / view["file"]).convert("RGBA")) / 255
            target = rgba[:, :, :3]
            image, alpha, _ = render(parameters, view, resolution)
            fitted = image.clamp(0, 1).cpu().numpy()
            alpha = alpha.cpu().numpy()
            target_alpha = rgba[:, :, 3:4]
            alpha_metrics[name] = {"outside_mass_fraction": float((alpha * (target_alpha < 0.05)).sum() / max(alpha.sum(), 1e-8)), "iou_0_2": float(((alpha > 0.2) & (target_alpha > 0.2)).sum() / max(((alpha > 0.2) | (target_alpha > 0.2)).sum(), 1))}
            structure = render_structure(vertices, pet.faces, view, resolution)[:, :, :3].clip(0, 1)
            for column, (label, pixels) in enumerate((("proxy", target), (appearance_label, fitted), ("structure", structure))):
                position = (column * resolution, row * (resolution + 24))
                draw.text((position[0] + 8, position[1] + 6), f"{name} {label}", fill="black")
                montage.paste(Image.fromarray((pixels * 255).round().astype(np.uint8)), (position[0], position[1] + 24))
            for column, (label, rgb, opacity) in enumerate((("proxy", target, target_alpha), (appearance_label, fitted, alpha))):
                position = (column * resolution, row * (resolution + 24))
                pixels = (rgb + (background - 1) * (1 - opacity)).clip(0, 1)
                viewer_draw.text((position[0] + 8, position[1] + 6), f"{name} {label}", fill="black")
                viewer_montage.paste(Image.fromarray((pixels * 255).round().astype(np.uint8)), (position[0], position[1] + 24))
    montage.save(output / "progress-montage.png")
    viewer_montage.save(output / "viewer-background-montage.png")
    core = vertices.cpu().numpy()
    proxy = np.load(directory / "proxy_mesh.npz")["vertices"]
    core_span = np.ptp(core, axis=0)
    proxy_span = np.ptp(proxy, axis=0)
    paws = core[[1330, 3282, 1521, 3473]]
    stats = {"stage": saved["stage"], "step": saved["step"], "count": len(parameters["means"]), "rest_max_span": float(core_span.max()), "core_span": core_span.tolist(), "proxy_span": proxy_span.tolist(), "core_span_ratio": (core_span / proxy_span).tolist(), "paw_nearest_proxy_distances": cKDTree(proxy).query(paws)[0].tolist(), "paw_min_y": float(paws[:, 1].min()), "proxy_min_y": float(proxy[:, 1].min()), "validation": saved["history"][-1] if saved["history"] else None}
    stats["alpha"] = alpha_metrics
    stats["appearance_source"] = str(Path(args.ply).resolve()) if args.ply else "checkpoint Gaussian state"
    held_out = []
    holdout_views = [view for view in configuration["views"] if view["split"] == "holdout"]
    heldout_montage = Image.new("RGB", (resolution * 3, (resolution + 24) * len(holdout_views)), tuple((background * 255).astype(int)))
    heldout_draw = ImageDraw.Draw(heldout_montage)
    with torch.no_grad():
        for row, view in enumerate(holdout_views):
            rgba = np.asarray(Image.open(directory / view["file"]).convert("RGBA")) / 255
            target = torch.as_tensor(rgba[:, :, :3], dtype=torch.float32, device="cuda")
            image, alpha, _ = render(parameters, view, resolution)
            before, before_alpha, _ = render(original_parameters, view, resolution) if args.ply else (image, alpha, None)
            a = alpha.cpu().numpy()
            b = before_alpha.cpu().numpy()
            target_alpha = rgba[:, :, 3:4]
            silhouette = a > 0.2
            original_silhouette = b > 0.2
            target_silhouette = target_alpha > 0.2
            item = {"view": view["id"], "psnr": float(-10 * (image - target).square().mean().clamp_min(1e-10).log10()), "ssim": float(ssim(image, target)), "before_psnr": float(-10 * (before - target).square().mean().clamp_min(1e-10).log10()), "before_ssim": float(ssim(before, target)), "silhouette_iou_before": float((silhouette & original_silhouette).sum() / max((silhouette | original_silhouette).sum(), 1)), "silhouette_iou_target": float((silhouette & target_silhouette).sum() / max((silhouette | target_silhouette).sum(), 1)), "silhouette_area_relative_change": float(silhouette.sum() / max(original_silhouette.sum(), 1) - 1), "outside_mass_fraction": float((a * (target_alpha < 0.05)).sum() / max(a.sum(), 1e-8))}
            held_out.append(item)
            for column, (label, rgb, opacity) in enumerate((("proxy", rgba[:, :, :3], target_alpha), ("before", before.cpu().numpy(), b), (appearance_label, image.cpu().numpy(), a))):
                position = (column * resolution, row * (resolution + 24))
                pixels = (rgb + (background - 1) * (1 - opacity)).clip(0, 1)
                heldout_draw.text((position[0] + 8, position[1] + 6), f"view {view['id']} {label}", fill="black")
                heldout_montage.paste(Image.fromarray((pixels * 255).round().astype(np.uint8)), (position[0], position[1] + 24))
    heldout_montage.save(output / "heldout-montage.png")
    stats["heldout_appearance"] = {"psnr": float(np.mean([item["psnr"] for item in held_out])), "ssim": float(np.mean([item["ssim"] for item in held_out])), "minimum_silhouette_iou_before": min(item["silhouette_iou_before"] for item in held_out), "maximum_absolute_area_change": max(abs(item["silhouette_area_relative_change"]) for item in held_out), "views": held_out}
    stats["finite_parameters"] = True
    stats["fit_complete"] = saved["stage"] == "free" and saved["step"] >= saved["args"]["free_steps"]
    stats["bound_iterations"] = saved["args"]["bound_steps"] if saved["stage"] == "free" else saved["step"]
    stats["free_iterations"] = saved["step"] if saved["stage"] == "free" else 0
    if (directory / "initialization.json").exists():
        initialization = json.loads((directory / "initialization.json").read_text())
        if "proxy_paw_medians" in initialization:
            anchors = np.asarray(initialization["proxy_paw_medians"])
            stats["paw_anchor_distances"] = np.linalg.norm(paws - anchors, axis=1).tolist()
            stats["paw_anchor_height_errors"] = (paws[:, 1] - anchors[:, 1]).tolist()
    (output / "progress.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
