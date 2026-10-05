"""Export the inspection skeleton and its sampled SMAL joint motion."""

import json
import struct
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def read_glb(path):
    data = Path(path).read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + json_length])
    binary = data[20 + json_length + 8:]

    def accessor(index):
        item = document["accessors"][index]
        view = document["bufferViews"][item["bufferView"]]
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[item["type"]]
        offset = view.get("byteOffset", 0) + item.get("byteOffset", 0)
        dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[item["componentType"]]
        return np.frombuffer(binary, dtype=dtype, count=item["count"] * width, offset=offset).reshape(-1, width).copy()
    return document, accessor


def load_baked_clips(path, pet):
    document, accessor = read_glb(path)
    primitive = document["meshes"][0]["primitives"][0]
    rest = pet().detach().cpu().numpy()
    if np.abs(accessor(primitive["attributes"]["POSITION"]) - rest).max() > 2e-5:
        raise RuntimeError("Baked skeleton rest mesh differs from the final fit")
    if not np.array_equal(accessor(primitive["indices"]).reshape(-1, 3), pet.faces.cpu().numpy()):
        raise RuntimeError("Baked skeleton topology differs from the final fit")
    pet.rest_joint_translations = np.asarray([node["translation"] for node in document["nodes"][1:36]], dtype=np.float32)
    pet.rest_joint_quaternions = np.asarray([node["rotation"] for node in document["nodes"][1:36]], dtype=np.float32)
    alignment = pet.alignment.cpu().numpy()
    device = pet.betas.device
    check_path = Path(path).with_name("bake-check.json")
    checks = json.loads(check_path.read_text()) if check_path.exists() else {}
    clips = {}
    for animation in document["animations"]:
        tracks = {}
        for channel in animation["channels"]:
            sampler = animation["samplers"][channel["sampler"]]
            times = accessor(sampler["input"])[:, 0]
            joint = channel["target"]["node"] - 1
            tracks[(joint, channel["target"]["path"])] = accessor(sampler["output"])
        translations = np.stack([tracks[(joint, "translation")] for joint in range(35)], axis=1)
        quaternions = np.stack([tracks[(joint, "rotation")] for joint in range(35)], axis=1)
        matrices = Rotation.from_quat(quaternions.reshape(-1, 4)).as_matrix().reshape(-1, 35, 3, 3)
        matrices[:, 0] = alignment.T @ matrices[:, 0]
        poses = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 35, 3)
        offsets = translations[:, 0] - pet.rest_joint_translations[0]
        with torch.no_grad():
            positions = np.stack([pet(torch.as_tensor(pose, device=device, dtype=torch.float32)[None], torch.as_tensor(offset, device=device, dtype=torch.float32)).cpu().numpy() for pose, offset in zip(poses, offsets)]).astype(np.float32)
        clips[animation["name"]] = {"duration": float(times[-1]), "times": times, "positions": positions, "jointTranslations": translations, "jointQuaternions": quaternions, "minimumPawHeight": float(positions[:, [1330, 3282, 1521, 3473], 1].min())}
        clips[animation["name"]]["maximumSoleOrientationErrorDegrees"] = checks.get("maximum_sole_orientation_error_degrees", {}).get(animation["name"], 0)
    return clips


def write_animated_glb(path, rest, faces, clips, pet):
    binary = bytearray()
    views = []
    accessors = []

    def accessor(value, kind, component=5126, bounds=False):
        dtype = "<u4" if component == 5125 else "<u2" if component == 5123 else "<f4"
        value = np.ascontiguousarray(value, dtype=dtype)
        binary.extend(b"\0" * ((-len(binary)) % 4))
        offset = len(binary)
        binary.extend(value.tobytes())
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": value.nbytes})
        item = {"bufferView": len(views) - 1, "componentType": component, "count": len(value), "type": kind}
        if bounds:
            item["min"] = value.min(axis=0).tolist()
            item["max"] = value.max(axis=0).tolist()
        accessors.append(item)
        return len(accessors) - 1

    parents = pet.smal.parents
    translations = pet.rest_joint_translations
    quaternions = pet.rest_joint_quaternions
    weights = pet.smal.weights.detach().cpu().numpy()
    influence_ids = np.argsort(-weights, axis=1)[:, :4]
    influences = np.take_along_axis(weights, influence_ids, axis=1)
    influences /= np.maximum(influences.sum(axis=1, keepdims=True), 1e-8)
    position = accessor(rest, "VEC3", bounds=True)
    indices = accessor(faces.reshape(-1), "SCALAR", 5125)
    joints_accessor = accessor(influence_ids, "VEC4", 5123)
    weights_accessor = accessor(influences, "VEC4")
    nodes = [{"name": "D-SMAL", "mesh": 0, "skin": 0}]
    world = []
    for joint in range(35):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(quaternions[joint]).as_matrix()
        matrix[:3, 3] = translations[joint]
        world.append(matrix if joint == 0 else world[int(parents[joint])] @ matrix)
        node = {"name": f"SMAL_joint_{joint:02d}", "translation": translations[joint].tolist(), "rotation": quaternions[joint].tolist()}
        children = [index + 1 for index in range(1, 35) if int(parents[index]) == joint]
        if children:
            node["children"] = children
        nodes.append(node)
    inverse_bind = accessor(np.stack([np.linalg.inv(matrix).T.reshape(16) for matrix in world]), "MAT4")
    animations = []
    for name, clip in clips.items():
        times = accessor(clip["times"], "SCALAR", bounds=True)
        samplers = []
        channels = []
        for joint in range(35):
            for key, kind, field in (("translation", "VEC3", "jointTranslations"), ("rotation", "VEC4", "jointQuaternions")):
                values = clip[field][:, joint].copy()
                if key == "rotation":
                    for frame in range(1, len(values)):
                        if np.dot(values[frame - 1], values[frame]) < 0:
                            values[frame] *= -1
                output = accessor(values, kind)
                channels.append({"sampler": len(samplers), "target": {"node": joint + 1, "path": key}})
                samplers.append({"input": times, "output": output, "interpolation": "LINEAR"})
        animations.append({"name": name, "samplers": samplers, "channels": channels})
    document = {"asset": {"version": "2.0", "generator": "SMAL-pets research reproduction"}, "extensionsUsed": ["KHR_materials_unlit"], "scene": 0, "scenes": [{"nodes": [0, 1]}], "nodes": nodes, "skins": [{"name": "D-SMAL 35-joint inspection rig", "inverseBindMatrices": inverse_bind, "skeleton": 1, "joints": list(range(1, 36))}], "meshes": [{"name": "D-SMAL structural proxy", "primitives": [{"attributes": {"POSITION": position, "JOINTS_0": joints_accessor, "WEIGHTS_0": weights_accessor}, "indices": indices, "material": 0}]}], "materials": [{"name": "Structural proxy", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0, "roughnessFactor": 1}, "extensions": {"KHR_materials_unlit": {}}, "doubleSided": True}], "animations": animations, "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors, "extras": {"appearanceDeformation": "Full D-SMAL baked vertices in manifest.json", "inspectionSkinInfluences": 4}}
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    content = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary))
    content += struct.pack("<I4s", len(encoded), b"JSON") + encoded
    content += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    Path(path).write_bytes(content)
