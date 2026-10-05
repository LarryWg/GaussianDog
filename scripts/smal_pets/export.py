"""Export ordered PLY splats, ten-face bindings and full vertex clips."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gaussians import progressive_order, read_ply, write_ply
from geometry import bind_faces, body_forward, deform
from model import PetModel
from bake import CLIPS, MOUTH_VERTEX, bake_animations, load_bake, save_bake
from glb import load_baked_clips, write_animated_glb


def gait_cadence(rest, clips):
    forward = body_forward(rest)
    result = {}
    for name in ("walk", "run"):
        clip = clips[name]
        paws = clip["positions"][:, [1330, 3282, 1521, 3473]]
        dt = np.diff(clip["times"])[:, None]
        velocities = np.einsum("flc,c->fl", np.diff(paws, axis=0), forward) / dt
        height = paws[:-1, :, 1]
        low = height <= np.quantile(height, 0.4, axis=0)[None]
        backwards = -velocities[low & (velocities < -1e-5)]
        if not len(backwards):
            raise RuntimeError(f"No stance velocity found for {name}")
        viewer_scale = 2.6 / np.ptp(rest, axis=0).max()
        result[name] = {"cyclesPerSecond": 1 / clip["duration"], "travelSpeed": float(np.median(backwards) * viewer_scale)}
    return result


def write_asset(output, parameters, rest, faces, clips, contact_errors=None, pet=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "clips").mkdir(exist_ok=True)
    face_ids, weights = bind_faces(parameters["means"], rest, faces)
    files = {"splats": "splats.ply", "restPositions": "rest.positions.bin", "faces": "faces.bin", "faceIds": "face-ids.bin", "weights": "face-weights.bin", "restTransforms": "rest-transforms.bin"}
    write_ply(output / files["splats"], parameters)
    np.asarray(rest, dtype="<f4").tofile(output / files["restPositions"])
    np.asarray(faces, dtype="<u4").tofile(output / files["faces"])
    face_ids.astype("<u2").tofile(output / files["faceIds"])
    weights.astype("<f4").tofile(output / files["weights"])
    quats = parameters["quats"][:, [1, 2, 3, 0]]
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    transforms = np.concatenate((quats, np.exp(parameters["log_scales"]), np.zeros((len(quats), 1))), axis=1)
    transforms.astype("<f4").tofile(output / files["restTransforms"])
    exported_clips = []
    for name, clip in clips.items():
        times = f"clips/{name}.times.bin"
        positions = f"clips/{name}.positions.bin"
        clip["times"].astype("<f4").tofile(output / times)
        clip["positions"].astype("<f4").tofile(output / positions)
        exported_clips.append({"name": name, "duration": clip["duration"], "times": times, "positions": positions})
    mouth_vertex = MOUTH_VERTEX if len(rest) == 3889 else 0
    mouth_candidates = np.where((faces == mouth_vertex).any(axis=1))[0]
    mouth_face = int(mouth_candidates[0])
    barycentric = [float(index == mouth_vertex) for index in faces[mouth_face]]
    cadence = gait_cadence(rest, clips) if len(rest) == 3889 else {name: {"cyclesPerSecond": 1 / clips[name]["duration"], "travelSpeed": 1.0} for name in ("walk", "run")}
    manifest = {"version": 1, "kind": "smal-pets-faces", "count": len(parameters["means"]), "vertexCount": len(rest), "faceCount": len(faces), "nearestFaces": 10, "coordinateSpace": "mesh-local-y-up", "topologyHash": hashlib.sha256(np.asarray(faces, dtype="<u4").tobytes()).hexdigest(), "files": files, "clips": exported_clips, "mouth": {"face": mouth_face, "barycentric": barycentric}, "gaitCadence": cadence}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sample = np.arange(min(4096, len(parameters["means"])))
    points = parameters["means"][sample]
    quaternions = parameters["quats"][sample]
    scales = np.exp(parameters["log_scales"][sample])
    posed, rotated, sized = deform(points, quaternions, scales, rest, rest, faces, face_ids[sample], weights[sample])
    identity_error = float(np.max(np.abs(posed - points)))
    if identity_error > 2e-6:
        raise RuntimeError(f"Face binding identity mismatch {identity_error}")
    if not np.allclose(np.linalg.norm(rotated, axis=1), 1, atol=1e-5):
        raise RuntimeError("Face binding produced invalid quaternions")
    if not np.allclose(sized, scales, atol=2e-6):
        raise RuntimeError("Face binding changed rest scales")
    viewer_scale = 2.6 / np.ptp(rest, axis=0).max()
    floor = float(rest[:, 1].min())
    full_mesh_heights = {name: float(clip["positions"][:, :, 1].min()) for name, clip in clips.items()}
    (output / "numerical-checks.json").write_text(json.dumps({"identity_max_error": identity_error, "weight_sum_max_error": float(np.abs(weights.sum(1) - 1).max()), "count": len(parameters["means"]), "sampleCount": len(points), "contact_max_errors": contact_errors or {}, "minimum_paw_heights": {name: clip.get("minimumPawHeight") for name, clip in clips.items()}, "minimum_full_mesh_heights": full_mesh_heights, "maximum_full_mesh_floor_penetration_viewer": {name: max(0, floor - height) * viewer_scale for name, height in full_mesh_heights.items()}, "vertexCount": len(rest), "full_smal": len(rest) == 3889}, indent=2) + "\n")
    if pet is not None:
        write_animated_glb(output / "dog-animated.glb", rest, faces, clips, pet)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bite-source", required=True)
    parser.add_argument("--bite-fit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ply")
    parser.add_argument("--baked-glb", help="Reuse inspected skeleton tracks without repeating contact optimization")
    parser.add_argument("--baked-npz", help="Reuse the exact solved full-SMAL motion buffers")
    parser.add_argument("--rebake-clips", nargs="+", choices=list(CLIPS), default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[50000, 100000, 150000])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if saved["stage"] != "free":
        raise RuntimeError("Export requires the free Gaussian stage")
    training_args = saved.get("args", {})
    if "free_steps" not in training_args or "bound_steps" not in training_args or int(saved["step"]) < int(training_args["free_steps"]):
        raise RuntimeError("Export requires a completed free Gaussian stage")
    pet = PetModel(args.bite_source, args.bite_fit, device=args.device)
    pet.load_state_dict(saved["pet"])
    faces = pet.faces.cpu().numpy()
    parameters = read_ply(args.ply) if args.ply else {name: value.cpu().numpy() for name, value in saved["gaussians"].items()}
    source_rest = pet().detach().cpu().numpy()
    forward = body_forward(source_rest)
    rotation = np.array([[-forward[2], 0, forward[0]], [0, 1, 0], [-forward[0], 0, -forward[2]]], dtype=np.float32)
    pet.set_alignment(rotation @ pet.alignment.detach().cpu().numpy(), float(pet.log_scale.detach().exp()), rotation @ pet.translation.detach().cpu().numpy())
    rest = pet().detach().cpu().numpy()
    nose_forward = rest[1863] - rest[452]
    nose_forward[1] = 0
    nose_forward /= max(np.linalg.norm(nose_forward), 1e-8)
    if nose_forward[2] >= 0:
        raise RuntimeError("Exported nose points behind the torso heading")
    parameters["means"] = (parameters["means"] @ rotation.T).astype(np.float32)
    quaternions = Rotation.from_matrix(rotation) * Rotation.from_quat(parameters["quats"][:, [1, 2, 3, 0]])
    parameters["quats"] = quaternions.as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    rotated_error = float(np.abs(rest - source_rest @ rotation.T).max())
    if rotated_error > 2e-6:
        raise RuntimeError(f"Export frame mismatch {rotated_error}")
    order = progressive_order(parameters)
    parameters = {name: value[order] for name, value in parameters.items()}
    if args.baked_npz and args.baked_glb:
        raise RuntimeError("Choose one solved motion source")
    if args.baked_npz:
        clips, contact_errors = load_bake(args.baked_npz, pet)
    elif args.baked_glb:
        clips = load_baked_clips(args.baked_glb, pet)
        checks = json.loads(Path(args.baked_glb).with_name("bake-check.json").read_text())
        contact_errors = checks["contact_max_errors"]
    else:
        if args.rebake_clips:
            raise RuntimeError("Selective baking requires a solved motion source")
        clips, contact_errors = bake_animations(pet)
    missing = set(CLIPS) - set(clips)
    if set(clips) - set(CLIPS) or missing - set(args.rebake_clips):
        raise RuntimeError("Solved motion must contain every profile not being rebaked")
    if args.rebake_clips:
        corrected_clips, corrected_errors = bake_animations(pet, names=args.rebake_clips)
        clips.update(corrected_clips)
        contact_errors.update(corrected_errors)
    clips = {name: clips[name] for name in CLIPS}
    viewer_scale = 2.6 / np.ptp(rest, axis=0).max()
    if max(contact_errors.values()) * viewer_scale >= 0.013:
        raise RuntimeError("Animation contact error exceeds 0.013 viewer units")
    floor = float(rest[[1330, 3282, 1521, 3473], 1].min())
    for name, clip in clips.items():
        if (floor - clip["minimumPawHeight"]) * viewer_scale >= 0.013:
            raise RuntimeError(f"Animation paw floor penetration in {name}")
        penetration = max(0, float(rest[:, 1].min()) - float(clip["positions"][:, :, 1].min())) * viewer_scale
        if penetration >= 0.013:
            raise RuntimeError(f"Animation full mesh floor penetration in {name}: {penetration}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_bake(Path(args.output) / "baked-clips.npz", rest, faces, clips, contact_errors, pet)
    write_asset(args.output, parameters, rest, faces, clips, contact_errors, pet)
    center = (rest.min(axis=0) + rest.max(axis=0)) / 2
    sniff_mouth = clips["sniff"]["positions"][:, MOUTH_VERTEX]
    export_provenance = {"source_checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), "bound_iterations": int(training_args["bound_steps"]), "free_iterations": int(saved["step"]), "requested_bound_iterations": int(training_args["bound_steps"]), "requested_free_iterations": int(training_args["free_steps"]), "fit_complete": True, "appearance_source": str(Path(args.ply).resolve()) if args.ply else "completed free Gaussian checkpoint", "count": len(parameters["means"]), "sniff_mouth_minimum_y": float(sniff_mouth[:, 1].min()), "sniff_mouth_minimum_height_above_floor_viewer": float((sniff_mouth[:, 1].min() - rest[:, 1].min()) * viewer_scale), "sniff_mouth_horizontal_forward_reach_viewer": float((-(sniff_mouth[:, 2] - center[2]) * viewer_scale).max()), "sniff_mouth_centered_viewer_bounds": [((sniff_mouth.min(axis=0) - center) * viewer_scale).tolist(), ((sniff_mouth.max(axis=0) - center) * viewer_scale).tolist()]}
    export_provenance["baked_motion_source"] = str(Path(args.baked_glb).resolve()) if args.baked_glb else "full contact bake"
    if args.baked_npz:
        export_provenance["baked_motion_source"] = str(Path(args.baked_npz).resolve())
    export_provenance["rebaked_clips"] = args.rebake_clips
    if args.ply:
        export_provenance["appearance_sha256"] = hashlib.sha256(Path(args.ply).read_bytes()).hexdigest()
        appearance_record = Path(args.ply).with_suffix(".json")
        if appearance_record.exists():
            shutil.copyfile(appearance_record, Path(args.output) / appearance_record.name)
        refinement_record = Path(args.ply).with_name("result.json")
        if refinement_record.exists():
            shutil.copyfile(refinement_record, Path(args.output) / "dge-result.json")
    if args.baked_npz:
        recovery_record = Path(args.baked_npz).with_name("bake-check.json")
        if not recovery_record.exists():
            recovery_record = Path(args.baked_npz).with_name("motion-recovery.json")
        if recovery_record.exists():
            recovered = json.loads(recovery_record.read_text())
            export_provenance["baked_motion_source_kind"] = recovered.get("recovery", {}).get("source", "solved full-SMAL buffers")
            shutil.copyfile(recovery_record, Path(args.output) / "motion-recovery.json")
    (Path(args.output) / "training-provenance.json").write_text(json.dumps(export_provenance, indent=2) + "\n")
    (Path(args.output) / "export-frame.json").write_text(json.dumps({"rotation": rotation.tolist(), "heading_source": "front paw center minus rear paw center", "forward_before": forward.tolist(), "forward_after": (rotation @ forward).tolist(), "nose_forward_after": nose_forward.tolist(), "mouth_vertex": MOUTH_VERTEX, "mouth_rest_position": rest[MOUTH_VERTEX].tolist(), "rest_max_span": float(np.ptp(rest, axis=0).max()), "viewer_scale": float(2.6 / np.ptp(rest, axis=0).max()), "rest_rotation_max_error": rotated_error}, indent=2) + "\n")
    densities = []
    for cap in args.counts:
        if cap > len(parameters["means"]):
            densities.append({"requested": cap, "available": False, "count": len(parameters["means"]), "manifest": "manifest.json"})
            continue
        count = min(cap, len(parameters["means"]))
        selected = {name: value[:count] for name, value in parameters.items()}
        path = Path(args.output) / f"density-{cap}"
        write_asset(path, selected, rest, faces, clips, contact_errors)
        densities.append({"requested": cap, "available": True, "count": count, "manifest": f"density-{cap}/manifest.json"})
    (Path(args.output) / "densities.json").write_text(json.dumps(densities, indent=2) + "\n")
    print(json.dumps({"export": args.output, "count": len(parameters["means"]), "densities": densities}), flush=True)


if __name__ == "__main__":
    main()
