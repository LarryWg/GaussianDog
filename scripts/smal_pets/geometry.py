"""SMAL-pets face frames and the browser deformation reference."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def normalize(value, epsilon=1e-8):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), epsilon)


def body_forward(vertices):
    forward = vertices[[1330, 3282]].mean(axis=0) - vertices[[1521, 3473]].mean(axis=0)
    forward[1] = 0
    return normalize(forward)


def face_frames(vertices, faces):
    triangles = vertices[faces]
    edge = triangles[:, 1] - triangles[:, 0]
    normal = np.cross(edge, triangles[:, 2] - triangles[:, 0])
    valid = (np.linalg.norm(edge, axis=1) > 1e-8) & (np.linalg.norm(normal, axis=1) > 1e-8)
    x_axis = normalize(edge)
    z_axis = normalize(normal)
    y_axis = np.cross(z_axis, x_axis)
    matrices = np.stack((x_axis, y_axis, z_axis), axis=-1)
    matrices[~valid] = np.eye(3)
    lengths = sum(np.linalg.norm(triangles[:, (i + 1) % 3] - triangles[:, i], axis=1) for i in range(3))
    return triangles.mean(axis=1), matrices, lengths, valid


def bind_faces(points, vertices, faces, neighbors=10):
    centers, _, _, _ = face_frames(vertices, faces)
    distances, indices = cKDTree(centers).query(points, k=neighbors)
    weights = 1 / np.maximum(distances, 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    return indices.astype(np.uint16), weights.astype(np.float32)


def deform(points, quaternions, scales, rest, posed, faces, indices, weights):
    """Quaternions use wxyz here, matching Gaussian PLY files."""
    rest_centers, rest_frames, rest_lengths, rest_valid = face_frames(rest, faces)
    centers, frames, lengths, valid = face_frames(posed, faces)
    delta = frames @ rest_frames.transpose(0, 2, 1)
    good = rest_valid & valid
    delta[~good] = np.eye(3)
    ratio = np.sqrt(lengths / np.maximum(rest_lengths, 1e-8))
    ratio[~good] = 1
    local = points[:, None] - rest_centers[indices]
    candidates = centers[indices] + np.einsum("nkij,nkj->nki", delta[indices], local)
    output = (candidates * weights[:, :, None]).sum(axis=1)
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    candidates_r = delta[indices] @ rotations[:, None]
    candidates_q = Rotation.from_matrix(candidates_r.reshape(-1, 3, 3)).as_quat().reshape(-1, indices.shape[1], 4)
    signs = np.where((candidates_q * candidates_q[:, :1]).sum(axis=-1) < 0, -1, 1)
    output_q = normalize((candidates_q * (weights * signs)[:, :, None]).sum(axis=1))
    output_s = scales * (ratio[indices] * weights).sum(axis=1, keepdims=True)
    return output, output_q[:, [3, 0, 1, 2]], output_s


def camera_views(count=96, resolution=512, radius=2.0, fov=40.0):
    focal = resolution / (2 * np.tan(np.deg2rad(fov) / 2))
    intrinsic = np.array([[focal, 0, resolution / 2], [0, focal, resolution / 2], [0, 0, 1]])
    result = []
    for index in range(count):
        y = 1 - 2 * (index + 0.5) / count
        angle = index * np.pi * (3 - np.sqrt(5))
        position = radius * np.array([np.cos(angle) * np.sqrt(1 - y * y), y, np.sin(angle) * np.sqrt(1 - y * y)])
        forward = normalize(-position)
        right = normalize(np.cross(forward, [0, 1, 0]))
        up = np.cross(right, forward)
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = np.stack((right, -up, forward), axis=-1)
        camera_to_world[:3, 3] = position
        result.append({"id": index, "file": f"images/{index:03d}.png", "split": "holdout" if index % 12 == 0 else "train", "K": intrinsic.tolist(), "view": np.linalg.inv(camera_to_world).tolist()})
    return result


def align_mesh(source, target):
    """Keep native Z upright and fit yaw, scale and translation with ICP."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    target_scale = np.max(np.ptp(target, axis=0))
    tree = cKDTree(target)
    candidates = []
    upright = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ upright
        scale = target_scale / max(np.ptp(source @ matrix.T, axis=0).max(), 1e-8)
        shifted = (source - source_center) @ matrix.T * scale + target_center
        distance, _ = tree.query(shifted)
        candidates.append((np.mean(np.minimum(distance, 0.1) ** 2), matrix, scale))
    _, rotation, scale = min(candidates, key=lambda value: value[0])
    translation = target_center - source_center @ rotation.T * scale
    for _ in range(30):
        aligned = source @ rotation.T * scale + translation
        distance, closest = tree.query(aligned)
        keep = distance < np.quantile(distance, 0.85)
        a = source[keep]
        b = target[closest[keep]]
        ac = a.mean(axis=0)
        bc = b.mean(axis=0)
        native = (a - ac) @ upright.T
        centered = b - bc
        cosine = (native[:, 0] * centered[:, 0] + native[:, 2] * centered[:, 2]).sum()
        sine = (native[:, 2] * centered[:, 0] - native[:, 0] * centered[:, 2]).sum()
        angle = np.arctan2(sine, cosine)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ upright
        scale = float((((a - ac) @ rotation.T) * centered).sum() / np.maximum(np.square(a - ac).sum(), 1e-8))
        translation = bc - ac @ rotation.T * scale
    return rotation, scale, translation
