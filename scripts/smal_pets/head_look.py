"""Export full-weight live gaze bindings from the completed SMAL fit."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from geometry import body_forward
from model import PetModel


EYES = [1068, 1080, 1029, 1226, 2660, 3030, 2675, 3038]


def globals_from_locals(translations, rotations, parents):
    result = []
    for joint, parent in enumerate(parents):
        matrix = np.eye(4)
        matrix[:3, :3] = rotations[joint]
        matrix[:3, 3] = translations[joint]
        result.append(matrix if parent < 0 else result[parent] @ matrix)
    return np.asarray(result)


def gaze_rotations(rotations, neutral_yaw, yaw, pitch, rest_rotations, yaw_axes, pitch_axes, neck_weight):
    result = rotations.copy()
    for index, joint in enumerate((15, 16)):
        fraction = neck_weight if index == 0 else 1 - neck_weight
        neutral = rest_rotations[joint] @ Rotation.from_rotvec(yaw_axes[index] * neutral_yaw * fraction).as_matrix() @ rest_rotations[joint].T
        result[joint] = neutral @ result[joint] @ Rotation.from_rotvec(yaw_axes[index] * yaw * fraction).as_matrix() @ Rotation.from_rotvec(pitch_axes[index] * pitch * fraction).as_matrix()
    return result


def heading(vertices):
    forward = vertices[1863] - vertices[EYES].mean(axis=0)
    return float(np.arctan2(-forward[0], -forward[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bite-source", required=True)
    parser.add_argument("--bite-fit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    pet = PetModel(args.bite_source, args.bite_fit, device="cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["stage"] != "free" or saved["step"] < saved["args"]["free_steps"]:
        raise RuntimeError("Head look bindings require the completed fit")
    pet.load_state_dict(saved["pet"])
    source_forward = body_forward(pet().detach().numpy())
    frame_rotation = np.array([[-source_forward[2], 0, source_forward[0]], [0, 1, 0], [-source_forward[0], 0, -source_forward[2]]], dtype=np.float32)
    pet.set_alignment(frame_rotation @ pet.alignment.numpy(), float(pet.log_scale.detach().exp()), frame_rotation @ pet.translation.detach().numpy())
    assets = Path(args.assets)
    manifest = json.loads((assets / "manifest.json").read_text())
    baked = np.load(assets / "baked-clips.npz")
    rest = np.fromfile(assets / manifest["files"]["restPositions"], dtype="<f4").reshape(-1, 3)
    parity = float(np.abs(pet().detach().numpy() - rest).max())
    if parity > 2e-6:
        raise RuntimeError(f"Fit rest differs from published mesh {parity}")
    faces = np.fromfile(assets / manifest["files"]["faces"], dtype="<u4")
    topology_hash = hashlib.sha256(faces.tobytes()).hexdigest()
    if topology_hash != manifest["topologyHash"]:
        raise RuntimeError("Published topology hash differs")
    parents = np.asarray(pet.smal.parents, dtype=int)
    weights = pet.smal.weights.detach().numpy().astype(np.float64)
    shaped = pet.smal.v_template + pet.offsets[0] + (pet.betas @ pet.smal.shapedirs[:30]).reshape(3889, 3)
    native_joints = (shaped.T @ pet.smal.J_regressor).T.detach().numpy()
    limb_scales = torch.exp(pet.limbs @ pet.smal.betas_scale_mask).reshape(1, 35, 3)
    rest_translations = baked["restJointTranslations"].astype(np.float64)
    rest_rotations = Rotation.from_quat(baked["restJointQuaternions"]).as_matrix()
    rest_globals = globals_from_locals(rest_translations, rest_rotations, parents)
    native_rotations = rest_rotations.copy()
    native_rotations[0] = pet.alignment.numpy().T @ native_rotations[0]
    from smal_pytorch.smal_model.batch_lbs import batch_global_rigid_transformation_biggs
    _, native_skin = batch_global_rigid_transformation_biggs(torch.tensor(native_rotations[None], dtype=torch.float32), torch.tensor(native_joints[None], dtype=torch.float32), parents, torch.diag_embed(limb_scales))
    canonical = np.eye(4)
    canonical[:3, :3] = pet.alignment.numpy() * float(pet.log_scale.detach().exp())
    canonical[:3, 3] = pet.translation.detach().numpy()
    bind = np.linalg.inv(rest_globals) @ canonical @ native_skin.detach().numpy()[0]
    bind = bind.astype(np.float32).astype(np.float64)
    blended_rest = np.einsum("vj,jab->vab", weights, rest_globals @ bind)
    up = np.array([0.0, 1.0, 0.0])
    skull_forward = rest[1863] - rest[EYES].mean(axis=0)
    skull_forward[1] = 0
    skull_forward /= np.linalg.norm(skull_forward)
    side = np.cross(skull_forward, up)
    yaw_axes = np.array([rest_globals[joint, :3, :3].T @ up for joint in (15, 16)])
    pitch_axes = np.array([rest_globals[joint, :3, :3].T @ side for joint in (15, 16)])
    neck_weight = 0.3

    def transfer(translations, rotations, points, neutral_yaw, yaw, pitch):
        before_globals = globals_from_locals(translations, rotations, parents)
        changed = gaze_rotations(rotations, neutral_yaw, yaw, pitch, rest_rotations, yaw_axes, pitch_axes, neck_weight)
        after_globals = globals_from_locals(translations, changed, parents)
        before = np.einsum("vj,jab->vab", weights, before_globals @ bind)
        after = np.einsum("vj,jab->vab", weights, after_globals @ bind)
        native = np.linalg.solve(before[:, :3, :3], (points - before[:, :3, 3])[..., None])[..., 0]
        return np.einsum("vab,vb->va", after[:, :3, :3], native) + after[:, :3, 3], changed

    initial_yaw = heading(rest)
    neutral_yaw = float(brentq(lambda yaw: heading(transfer(rest_translations, rest_rotations, rest, yaw, 0, 0)[0]), -initial_yaw - 0.4, -initial_yaw + 0.4, xtol=1e-12))
    neutral_rest, _ = transfer(rest_translations, rest_rotations, rest, neutral_yaw, 0, 0)
    neutral_rotations = [Rotation.from_matrix(rest_rotations[joint] @ Rotation.from_rotvec(yaw_axes[index] * neutral_yaw * (neck_weight if index == 0 else 1 - neck_weight)).as_matrix() @ rest_rotations[joint].T).as_quat().tolist() for index, joint in enumerate((15, 16))]
    errors = {}
    determinant = float(np.linalg.det(blended_rest[:, :3, :3]).min())
    fixtures = []
    for name, frame in (("idle", 0), ("walk", 13), ("sniff", 26), ("jump", 45), ("sit", 75), ("bark", 25)):
        translations = baked[f"{name}_jointTranslations"][frame].astype(np.float64)
        rotations = Rotation.from_quat(baked[f"{name}_jointQuaternions"][frame]).as_matrix()
        points = baked[f"{name}_positions"][frame].astype(np.float64)
        displayed, changed = transfer(translations, rotations, points, neutral_yaw, 0.2, 0.1)
        native_pose = changed.copy()
        native_pose[0] = pet.alignment.numpy().T @ native_pose[0]
        pose = torch.tensor(Rotation.from_matrix(native_pose).as_rotvec()[None], dtype=torch.float32)
        offset = torch.tensor(translations[0] - rest_translations[0], dtype=torch.float32)
        expected = pet(pose, offset).detach().numpy()
        errors[name] = float(np.abs(displayed - expected).max())
        if errors[name] > 3e-6:
            raise RuntimeError(f"Full SMAL gaze parity failed for {name} {errors[name]}")
        fixtures.append({"name": name, "frame": frame, "yaw": 0.2, "pitch": 0.1, "maximumError": errors[name]})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(output / "head-look.weights.bin")
    bind.transpose(0, 2, 1).astype("<f4").tofile(output / "head-look.bind-matrices.bin")
    data = {"version": 1, "vertexCount": len(rest), "topologyHash": topology_hash, "jointNames": [f"SMAL_joint_{joint:02d}" for joint in range(35)], "parents": parents.tolist(), "neck": 15, "head": 16, "neckWeight": neck_weight, "neutralYaw": neutral_yaw, "neutralRotations": neutral_rotations, "yawAxes": yaw_axes.tolist(), "pitchAxes": pitch_axes.tolist(), "files": {"weights": "head-look.weights.bin", "bindMatrices": "head-look.bind-matrices.bin"}}
    (output / "head-look.json").write_text(json.dumps(data, indent=2) + "\n")
    report = {"status": "passed", "fullJointCount": 35, "affectedVertices": int((weights[:, [15, 16, 32, 33, 34]].sum(axis=1) > 0).sum()), "restMaximumError": parity, "originalHeadYaw": initial_yaw, "neutralYaw": neutral_yaw, "centeredHeadYaw": heading(neutral_rest), "minimumRestSkinDeterminant": determinant, "fullSMALMaximumErrors": errors, "poseCorrectiveBasisNorm": float(pet.smal.posedirs.norm()), "sourceCheckpoint": str(Path(args.checkpoint).resolve()), "sourceCheckpointSHA256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), "sourceBakedMotion": str((assets / "baked-clips.npz").resolve()), "sourceBakedMotionSHA256": hashlib.sha256((assets / "baked-clips.npz").read_bytes()).hexdigest(), "fixtures": fixtures}
    (output / "head-look-check.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.update_manifest:
        if output.resolve() != assets.resolve():
            raise RuntimeError("Manifest metadata must be beside the appearance files")
        manifest["files"]["headLook"] = "head-look.json"
        (assets / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
