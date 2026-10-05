"""Bake action profiles with full D-SMAL deformation."""

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from model import axis_angle_matrix
from geometry import body_forward

CLIPS = {"idle": 90, "walk": 36, "run": 15, "sit": 75, "jump": 72, "bark": 72, "paw": 90, "spin": 120, "playbow": 75, "sniff": 90, "dig": 60, "wag": 60}
LEGS = [(7, 8, 9, 1330), (11, 12, 13, 3282), (17, 18, 19, 1521), (21, 22, 23, 3473)]
MOUTH_VERTEX = 910
SOLES = [[1399, 1349, 1319], [3352, 3261, 3324], [1628, 1684, 1668], [3549, 3488, 3621]]


def smooth(start, end, value):
    x = np.clip((value - start) / (end - start), 0, 1)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def window(start, rise, fall, end, value):
    return smooth(start, rise, value) * (1 - smooth(fall, end, value))


def pulse(start, peak, end, value):
    return window(start, peak, peak, end, value)


def save_bake(path, rest, faces, clips, errors, pet):
    data = {"rest": rest, "faces": faces, "names": np.asarray(list(clips)), "contactErrors": np.asarray([errors[name] for name in clips]), "restJointTranslations": pet.rest_joint_translations, "restJointQuaternions": pet.rest_joint_quaternions}
    for name, clip in clips.items():
        for field in ("times", "positions", "jointTranslations", "jointQuaternions", "minimumPawHeight", "maximumSoleOrientationErrorDegrees"):
            data[f"{name}_{field}"] = clip[field]
    np.savez(path, **data)


def load_bake(path, pet):
    saved = np.load(path)
    rest = pet().detach().cpu().numpy()
    if np.max(np.abs(saved["rest"] - rest)) > 2e-6 or not np.array_equal(saved["faces"], pet.faces.cpu().numpy()):
        raise RuntimeError("Solved motion does not match the completed fit")
    pet.rest_joint_translations = saved["restJointTranslations"]
    pet.rest_joint_quaternions = saved["restJointQuaternions"]
    clips = {}
    names = saved["names"].tolist()
    for name in names:
        clip = {field: saved[f"{name}_{field}"] for field in ("times", "positions", "jointTranslations", "jointQuaternions")}
        clip["duration"] = len(clip["times"][1:]) / 30
        clip["minimumPawHeight"] = float(saved[f"{name}_minimumPawHeight"])
        clip["maximumSoleOrientationErrorDegrees"] = float(saved[f"{name}_maximumSoleOrientationErrorDegrees"])
        if not all(np.isfinite(value).all() for value in clip.values()):
            raise RuntimeError(f"Non-finite solved clip {name}")
        if clip["positions"].shape != (CLIPS[name] + 1, 3889, 3):
            raise RuntimeError(f"Incorrect solved clip dimensions {name}")
        clips[name] = clip
    return clips, dict(zip(names, saved["contactErrors"].tolist()))


def bake_animations(pet, fps=30, names=None):
    trainable = [parameter.requires_grad for parameter in pet.parameters()]
    for parameter in pet.parameters():
        parameter.requires_grad_(False)
    device = pet.betas.device
    rest = pet().detach().cpu().numpy()
    initial = axis_angle_matrix(pet.pose).detach().cpu().numpy()[0]
    alignment = pet.alignment.detach().cpu().numpy()
    parents = pet.smal.parents
    head = int(parents[32])
    neck = int(parents[head])
    torso = []
    ancestor = int(parents[neck])
    while ancestor > 0:
        torso.append(ancestor)
        ancestor = int(parents[ancestor])
    torso = torso[::-1] or [1, 2, 3]
    forward = body_forward(rest)
    up = np.array([0.0, 1.0, 0.0])
    side = np.cross(forward, up)
    bases = []
    for joint in range(35):
        parent_rotation = np.eye(3) if joint == 0 else bases[int(parents[joint])]
        bases.append(parent_rotation @ initial[joint])
    local_side = np.array([(alignment @ rotation).T @ side for rotation in bases])
    local_up = np.array([(alignment @ rotation).T @ up for rotation in bases])
    targets = rest[[leg[3] for leg in LEGS]]
    floor_height = float(rest[:, 1].min())
    sole_ids = torch.as_tensor(SOLES, device=device, dtype=torch.long)

    def sole_frames(vertices):
        triangles = vertices[sole_ids]
        x = torch.nn.functional.normalize(triangles[:, 1] - triangles[:, 0], dim=-1)
        z = torch.nn.functional.normalize(torch.cross(x, triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        return torch.stack((x, y, z), dim=-1)

    foot_frames = sole_frames(torch.as_tensor(rest, device=device, dtype=torch.float32))

    def joint_pose(rotations, offset):
        world_rotations = []
        for joint in range(35):
            parent_rotation = alignment if joint == 0 else world_rotations[int(parents[joint])]
            world_rotations.append(parent_rotation @ rotations[joint])
        native_joints = pet.smal.J_transformed.detach().cpu().numpy()[0]
        joint_positions = native_joints @ alignment.T * float(pet.log_scale.detach().exp()) + pet.translation.detach().cpu().numpy() + offset
        local_positions = joint_positions.copy()
        local_rotations = rotations.copy()
        local_rotations[0] = world_rotations[0]
        for joint in range(1, 35):
            parent = int(parents[joint])
            local_positions[joint] = world_rotations[parent].T @ (joint_positions[joint] - joint_positions[parent])
        return local_positions.astype(np.float32), Rotation.from_matrix(local_rotations).as_quat().astype(np.float32)

    rest_joint_translations, rest_joint_quaternions = joint_pose(initial, np.zeros(3))
    pet.rest_joint_translations = rest_joint_translations
    pet.rest_joint_quaternions = rest_joint_quaternions
    clips = {}
    contact_errors = {}
    for name, length in CLIPS.items():
        if names is not None and name not in names:
            continue
        frames = []
        joint_translations = []
        joint_quaternions = []
        errors = []
        sole_angles = []
        previous_correction = None
        previous_joint_ids = None
        for frame in range(length + 1):
            if frame == length and name != "sit":
                frames.append(frames[0].copy())
                joint_translations.append(joint_translations[0].copy())
                joint_quaternions.append(joint_quaternions[0].copy())
                continue
            t = frame / length
            rotations = initial.copy()
            translation = np.zeros(3)
            paw_offsets = np.zeros((4, 3))

            def rotate(joint, angle, axis=None):
                vector = local_side[joint] if axis is None else axis[joint]
                rotations[joint] = rotations[joint] @ Rotation.from_rotvec(vector * angle).as_matrix()

            if name in ("walk", "run", "spin"):
                running = name == "run"
                amount = smooth(0, 0.12, t) * (1 - smooth(0.84, 1, t)) if name == "spin" else 1
                offsets = [0, 0.06, 0.5, 0.56] if running else [0, 0.5, 0.75, 0.25]
                for index, (upper, lower, foot, _) in enumerate(LEGS):
                    swing = math.sin(2 * math.pi * (t * (3 if name == "spin" else 1) + offsets[index]))
                    amplitude = 0.48 if running else 0.28
                    if name == "spin":
                        amplitude = 0.38 if index % 2 == 0 else 0.18
                    rotate(upper, amount * swing * amplitude)
                    rotate(lower, amount * max(0, -swing) * (0.55 if running else 0.32))
                    rotate(foot, -amount * swing * (0.15 if running else 0.10))
                translation += up * (0.045 if running else 0.012) * (1 - math.cos(4 * math.pi * t))
                rotate(head, math.sin(4 * math.pi * t) * 0.035)
                if name == "spin":
                    turn = smooth(0, 1, t) * 2 * math.pi
                    rotate(0, turn, local_up)
                    center = pet.translation.detach().cpu().numpy()
                    translation += Rotation.from_rotvec(up * turn).apply(center) - center
            elif name == "idle":
                rotate(head, math.sin(2 * math.pi * t) * 0.035)
            elif name == "sit":
                settle = smooth(0, 0.55, t)
                translation += (-0.28 * up - 0.05 * forward) * settle
                rotate(torso[0], 0.45 * settle)
                for upper, lower, _, _ in LEGS[2:]:
                    rotate(upper, 1.0 * settle)
                    rotate(lower, -1.5 * settle)
                rotate(head, -0.38 * settle)
                rotate(25, -0.65 * settle)
            elif name == "jump":
                crouch = pulse(0, 0.12, 0.27, t)
                absorb = pulse(0.69, 0.84, 1, t)
                flight = pulse(0.28, 0.51, 0.70, t)
                pitch = 0.14 * pulse(0.15, 0.31, 0.53, t) - 0.15 * pulse(0.50, 0.71, 0.96, t)
                translation += up * (0.20 * flight - 0.035 * crouch - 0.04 * absorb)
                rotate(0, pitch)
                rotate(neck, -0.6 * pitch)
                rotate(head, -0.2 * pitch)
                for index, (upper, lower, _, _) in enumerate(LEGS):
                    front = index < 2
                    lift = pulse(0.15, 0.43, 0.73, t) if front else pulse(0.28, 0.51, 0.86, t)
                    paw_offsets[index] = up * (0.30 if front else 0.28) * lift + forward * (0.035 if front else 0.012) * lift
                    rotate(upper, (-0.28 if front else 0.22) * lift)
                    rotate(lower, (0.55 if front else -0.42) * lift + (0.18 if front else -0.18) * crouch)
            elif name == "bark":
                bark = max(0, math.sin(6 * math.pi * t)) ** 2
                rotate(head, -0.23 * bark)
                rotate(32, -0.55 * bark)
                rotate(neck, -0.07 * bark)
                translation += forward * -0.025 * bark
            elif name == "paw":
                lift = window(0.05, 0.25, 0.72, 0.92, t)
                rotate(7, 1.20 * lift)
                rotate(8, -0.90 * lift)
                rotate(9, 0.28 * math.sin(6 * math.pi * t) * lift)
                rotate(head, 0.10 * lift, local_up)
                translation += (-side * 0.018 - up * 0.008) * lift
            elif name == "playbow":
                bow = window(0.05, 0.28, 0.72, 0.96, t)
                translation += up * -0.10 * bow
                for joint in torso:
                    rotate(joint, -0.23 * bow / len(torso))
                for index, (upper, lower, _, _) in enumerate(LEGS):
                    rotate(lower if index < 2 else upper, (0.56 if index < 2 else -0.16) * bow)
                rotate(head, -0.16 * bow)
            elif name == "sniff":
                translation += up * -0.115
                for joint in torso:
                    rotate(joint, -0.65 / len(torso))
                rotate(neck, -0.75)
                rotate(head, 0.1)
                rotate(head, -0.6, local_up)
                rotate(32, -0.2)
            elif name == "dig":
                translation += -up * 0.045 + side * (0.006 * math.cos(4 * math.pi * t))
                rotate(0, -0.10)
                for joint in torso:
                    rotate(joint, -0.22 / len(torso))
                rotate(neck, -0.30)
                rotate(head, 0.05)
                rotate(head, 0.025 * math.sin(4 * math.pi * t), local_up)
                for index, (upper, lower, _, _) in enumerate(LEGS[:2]):
                    phase = (2 * t + index * 0.5) % 1
                    if phase <= 0.55:
                        reach = 0.07 - 0.13 * smooth(0, 0.55, phase)
                        lift = 0
                    else:
                        recovery = (phase - 0.55) / 0.45
                        reach = -0.06 + 0.13 * smooth(0, 1, recovery)
                        lift = math.sin(math.pi * recovery) ** 2
                    paw_offsets[index] = forward * reach + up * 0.075 * lift
                    rotate(upper, -0.22 * lift)
                    rotate(lower, 0.40 * lift)
            elif name == "wag":
                rotate(head, 0.04 * math.sin(2 * math.pi * t))
            looping = name in {"idle", "walk", "run", "sniff", "dig", "wag"}
            envelope = 1 if looping else window(0, 0.12, 0.86, 1, t)
            breath = (1 - math.cos(2 * math.pi * t)) * 0.5
            if name in {"idle", "wag"}:
                for index, joint in enumerate(torso):
                    rotate(joint, (0.004 if index < 2 else -0.004) * breath)
                rotate(neck, math.sin(2 * math.pi * t) * 0.012)
            tail_weights = np.array([0.48 ** index for index in range(7)])
            tail_weights /= tail_weights.sum()
            frequency = 5 if name in {"wag", "bark", "playbow"} else 2
            amplitude = 0.48 if frequency == 5 else 0.14
            for index, joint in enumerate(range(25, 32)):
                lag = index * 0.32
                rotate(joint, envelope * amplitude * tail_weights[index] * (math.sin(2 * math.pi * frequency * t - lag) + math.sin(lag)), local_up)
            for index, joint in enumerate((33, 34)):
                lag = 0.4 + index * 0.18
                rotate(joint, envelope * (math.sin(4 * math.pi * t - lag) + math.sin(lag)) * (0.035 if name == "jump" else 0.018))
            pose = torch.as_tensor(Rotation.from_matrix(rotations).as_rotvec(), device=device, dtype=torch.float32)[None]
            offset = torch.as_tensor(translation, device=device, dtype=torch.float32)
            contact = list(range(4)) if name in {"sit", "playbow", "sniff", "jump", "dig"} else [1, 2, 3] if name == "paw" else []
            if contact:
                joint_ids = [joint for index in contact for joint in LEGS[index][:3]]
                point_ids = [LEGS[index][3] for index in contact]
                goal = targets[contact].copy()
                marker_ids = np.array([[LEGS[index][3], *SOLES[index]] for index in contact]).reshape(-1)
                marker_goal = rest[marker_ids].copy().reshape(len(contact), 4, 3)
                goal += paw_offsets[contact]
                marker_goal += paw_offsets[contact, None, :]
                if name == "sit":
                    for index, leg in enumerate(contact):
                        if leg >= 2:
                            goal[index] += forward * 0.14 * settle
                            marker_goal[index] += forward * 0.14 * settle
                goal = torch.as_tensor(goal, device=device, dtype=torch.float32)
                marker_goal = torch.as_tensor(marker_goal.reshape(-1, 3), device=device, dtype=torch.float32)
                warm = previous_joint_ids == joint_ids and previous_correction is not None
                seed = previous_correction.clone().requires_grad_(True) if warm else torch.zeros((len(joint_ids), 3), device=device, requires_grad=True)
                mask = torch.nn.functional.one_hot(torch.as_tensor(joint_ids, device=device), 35).float()
                if not warm:
                    seed_optimizer = torch.optim.LBFGS([seed], lr=1.0, max_iter=40, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-10)
                    def seed_closure():
                        seed_optimizer.zero_grad()
                        corrected = pose + (mask.T @ seed)[None]
                        loss = (pet(corrected, offset)[point_ids] - goal).square().mean() + 0.000001 * seed.square().mean()
                        loss.backward()
                        return loss
                    seed_optimizer.step(seed_closure)
                base_pose = (pose + (mask.T @ seed.detach())[None]).detach()
                correction = torch.zeros((len(joint_ids), 3), device=device, requires_grad=True)
                optimizer = torch.optim.LBFGS([correction], lr=1.0, max_iter=300, history_size=25, line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-12)
                def closure():
                    optimizer.zero_grad()
                    corrected = base_pose + (mask.T @ correction)[None]
                    vertices = pet(corrected, offset)
                    positions_loss = (vertices[marker_ids] - marker_goal).square().mean()
                    orientation_loss = (sole_frames(vertices)[contact] - foot_frames[contact]).square().mean()
                    floor_loss = torch.relu(floor_height - vertices[:, 1]).square().mean()
                    loss = positions_loss + 0.005 * orientation_loss + 100 * floor_loss + 1e-7 * correction.square().mean()
                    loss.backward()
                    return loss
                optimizer.step(closure)
                previous_correction = seed.detach() + correction.detach()
                previous_joint_ids = joint_ids
                pose = (base_pose + (mask.T @ correction.detach())[None]).detach()
            with torch.no_grad():
                positions = pet(pose, offset).cpu().numpy().astype(np.float32)
                if name in {"walk", "run", "spin", "jump"}:
                    lift = max(0, floor_height - float(positions[:, 1].min()))
                    translation += up * lift
                    offset = torch.as_tensor(translation, device=device, dtype=torch.float32)
                    positions = pet(pose, offset).cpu().numpy().astype(np.float32)
                final_rotations = axis_angle_matrix(pose).cpu().numpy()[0]
            frames.append(positions)
            translations, quaternions = joint_pose(final_rotations, translation)
            joint_translations.append(translations)
            joint_quaternions.append(quaternions)
            if contact:
                errors.append(float(np.linalg.norm(positions[marker_ids] - marker_goal.cpu().numpy(), axis=1).max()))
                posed_frames = sole_frames(torch.as_tensor(positions, device=device))[contact].detach().cpu().numpy()
                rest_frames = foot_frames[contact].cpu().numpy()
                relative = posed_frames @ rest_frames.transpose(0, 2, 1)
                cos_angle = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
                sole_angles.append(float(np.rad2deg(np.arccos(cos_angle)).max()))
            if frame and frame % 15 == 0:
                print(f"Baking {name} frame {frame}/{length}, target error {max(errors, default=0):.8f}", flush=True)
        frames = np.stack(frames)
        assert np.isfinite(frames).all(), name
        if name in {"idle", "walk", "run", "wag", "spin", "jump", "bark", "paw", "playbow", "dig"}:
            assert np.max(np.abs(frames[0] - frames[-1])) < 1e-4, name
        clips[name] = {"duration": length / fps, "times": np.arange(length + 1, dtype=np.float32) / fps, "positions": frames, "jointTranslations": np.stack(joint_translations), "jointQuaternions": np.stack(joint_quaternions), "minimumPawHeight": float(frames[:, [leg[3] for leg in LEGS], 1].min()), "maximumSoleOrientationErrorDegrees": max(sole_angles, default=0)}
        contact_errors[name] = max(errors, default=0)
        penetration = max(0, floor_height - float(frames[:, :, 1].min()))
        print(f"Baked {name} with {len(frames)} frames, contact {contact_errors[name]:.8f}, floor penetration {penetration:.8f}", flush=True)
    for parameter, requires_grad in zip(pet.parameters(), trainable):
        parameter.requires_grad_(requires_grad)
    return clips, contact_errors
