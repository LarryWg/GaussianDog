"""Bake and inspect the canonical rig before expensive appearance training."""

import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
import torch
from model import PetModel
from bake import LEGS, MOUTH_VERTEX, SOLES, bake_animations, save_bake, smooth
from export import gait_cadence
from geometry import body_forward
from glb import load_baked_clips, write_animated_glb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bite-source", required=True)
    parser.add_argument("--bite-fit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Inspect a copied fitted checkpoint without publishing a final asset")
    parser.add_argument("--baked-glb", help="Recover full-SMAL clips from solved skeleton tracks")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fit = np.load(args.bite_fit)
    pet = PetModel(args.bite_source, args.bite_fit)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        pet.load_state_dict(saved["pet"])
        forward = body_forward(pet().detach().cpu().numpy())
        rotation = np.array([[-forward[2], 0, forward[0]], [0, 1, 0], [-forward[0], 0, -forward[2]]])
        pet.set_alignment(rotation @ pet.alignment.detach().cpu().numpy(), float(pet.log_scale.detach().exp()), rotation @ pet.translation.detach().cpu().numpy())
    else:
        transform = fit["coordinate_transform"]
        scale = np.linalg.norm(transform[:3, 0])
        pet.set_alignment(transform[:3, :3] / scale, scale, transform[:3, 3])
    rest = pet().detach().cpu().numpy()
    faces = pet.faces.cpu().numpy()
    recovery = {}
    if args.baked_glb:
        source = Path(args.baked_glb).parent
        source_checks = json.loads((source / "bake-check.json").read_text())
        clips = load_baked_clips(args.baked_glb, pet)
        contact_errors = {}
        errors = {}
        for name, clip in clips.items():
            frame = -1 if name == "sit" else len(clip["times"]) // 2
            if name in {"walk", "run", "spin"}:
                frame = len(clip["times"]) // 4
            original = trimesh.load(source / f"{name}.obj", process=False, maintain_order=True)
            errors[name] = float(np.abs(clip["positions"][frame] - np.asarray(original.vertices)).max())
            contact_error = 0
            for index, vertices in enumerate(clip["positions"]):
                t = index / (len(clip["times"]) - 1)
                flight = max(0, np.sin(np.pi * np.clip((t - 0.22) / 0.60, 0, 1))) ** 1.15
                contact = list(range(4)) if name in {"sit", "playbow", "sniff"} or name == "jump" and flight < 1e-5 else [1, 2, 3] if name == "paw" else []
                for leg in contact:
                    markers = [LEGS[leg][3], *SOLES[leg]]
                    goal = rest[markers].copy()
                    if name == "sit" and leg >= 2:
                        goal += body_forward(rest) * 0.14 * smooth(0, 0.55, t)
                    contact_error = max(contact_error, float(np.linalg.norm(vertices[markers] - goal, axis=1).max()))
            contact_errors[name] = contact_error
        assert max(errors.values()) < 2e-5, errors
        contact_drift = {name: abs(contact_errors[name] - source_checks["contact_max_errors"][name]) for name in clips}
        floor_drift = {name: abs(float(clip["positions"][:, :, 1].min()) - source_checks["minimum_full_mesh_heights"][name]) for name, clip in clips.items()}
        assert max(contact_drift.values()) < 2e-5, contact_drift
        assert max(floor_drift.values()) < 2e-5, floor_drift
        recovery = {"source": "decoded solved GLB tracks", "glb": str(Path(args.baked_glb).resolve()), "saved_obj_max_errors": errors, "contact_metric_max_drift": contact_drift, "full_floor_max_drift": floor_drift}
    else:
        clips, contact_errors = bake_animations(pet)
    save_bake(output / "baked-clips.npz", rest, faces, clips, contact_errors, pet)
    floor = float(rest[[1330, 3282, 1521, 3473], 1].min())
    viewer_scale = 2.6 / np.ptp(rest, axis=0).max()
    write_animated_glb(output / "dog-animated.glb", rest, faces, clips, pet)
    reconstructed = load_baked_clips(output / "dog-animated.glb", pet)
    roundtrip_errors = {name: float(np.abs(clips[name]["positions"] - reconstructed[name]["positions"]).max()) for name in clips}
    assert max(roundtrip_errors.values()) < 2e-5, roundtrip_errors
    pet.export_fit(output / "motion_fit.npz")
    for name, clip in clips.items():
        frame = -1 if name == "sit" else len(clip["times"]) // 2
        if name in {"walk", "run", "spin"}:
            frame = len(clip["times"]) // 4
        trimesh.Trimesh(vertices=clip["positions"][frame], faces=faces, process=False).export(output / f"{name}.obj")
    center = (rest.min(axis=0) + rest.max(axis=0)) / 2
    jaw = clips["sniff"]["positions"][:, MOUTH_VERTEX]
    full_heights = {name: float(clip["positions"][:, :, 1].min()) for name, clip in clips.items()}
    checks = {"status": "baked", "stage": "fitted_preflight" if args.checkpoint else "canonical_preflight", "vertexCount": len(rest), "jointCount": 35, "contact_max_errors": contact_errors, "contact_max_viewer_error": max(contact_errors.values()) * viewer_scale, "rest_paw_floor": floor, "gaitCadence": gait_cadence(rest, clips), "minimum_paw_heights": {name: clip["minimumPawHeight"] for name, clip in clips.items()}, "minimum_full_mesh_heights": full_heights, "full_mesh_floor_penetration_viewer": {name: max(0, float(rest[:, 1].min()) - height) * viewer_scale for name, height in full_heights.items()}, "sniff_jaw_height_above_floor_viewer": float((jaw[:, 1].min() - rest[:, 1].min()) * viewer_scale), "sniff_jaw_forward_reach_viewer": float((-(jaw[:, 2] - center[2]) * viewer_scale).max()), "sniff_jaw_lateral_range_viewer": ((jaw[:, 0] - center[0]) * viewer_scale).tolist(), "clip_durations": {name: clip["duration"] for name, clip in clips.items()}, "loop_seams": {name: float(np.abs(clip["positions"][0] - clip["positions"][-1]).max()) for name, clip in clips.items() if name != "sit"}}
    checks["maximum_sole_orientation_error_degrees"] = {name: clip["maximumSoleOrientationErrorDegrees"] for name, clip in clips.items()}
    checks["contact_metric"] = "maximum point distance across paw anchors and sole triplets"
    checks["skeleton_roundtrip_max_vertex_errors"] = roundtrip_errors
    checks["recovery"] = recovery
    checks["status"] = "passed" if checks["contact_max_viewer_error"] < 0.013 and max(checks["full_mesh_floor_penetration_viewer"].values()) < 0.013 else "failed"
    clip = clips["sniff"]
    before = np.searchsorted(clip["times"], 0.85) - 1
    fraction = (0.85 - clip["times"][before]) / (clip["times"][before + 1] - clip["times"][before])
    sample = (1 - fraction) * clip["positions"][before] + fraction * clip["positions"][before + 1]
    restored_sample = (1 - fraction) * reconstructed["sniff"]["positions"][before] + fraction * reconstructed["sniff"]["positions"][before + 1]
    checks["sniff_pickup_sample"] = {"time": 0.85, "mouth_vertex": MOUTH_VERTEX, "mouth_viewer": ((sample[MOUTH_VERTEX] - center) * viewer_scale + np.array([0, (center[1] - rest[:, 1].min()) * viewer_scale, 0])).tolist(), "skeleton_roundtrip_max_error": float(np.abs(sample - restored_sample).max())}
    (output / "bake-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks), flush=True)
    assert max(contact_errors.values()) * viewer_scale < 0.013, contact_errors
    assert max(checks["full_mesh_floor_penetration_viewer"].values()) < 0.013, checks["full_mesh_floor_penetration_viewer"]


if __name__ == "__main__":
    main()
