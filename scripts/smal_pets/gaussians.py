"""Read and write the standard Gaussian PLY parameterization."""

from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def progressive_order(parameters):
    points = parameters["means"]
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, 1e-8)
    cells = np.clip(((points - minimum) / span * 32).astype(np.int32), 0, 31)
    cell_ids = cells[:, 0] + 32 * cells[:, 1] + 1024 * cells[:, 2]
    opacity = 1 / (1 + np.exp(-parameters["opacity_logits"]))
    grouped = np.lexsort((-opacity, cell_ids))
    _, starts, counts = np.unique(cell_ids[grouped], return_index=True, return_counts=True)
    ranks = np.arange(len(points)) - np.repeat(starts, counts)
    cell_order = np.repeat(np.random.default_rng(42).permutation(len(starts)), counts)
    return grouped[np.lexsort((cell_order, ranks))]


def read_ply(path):
    data = PlyData.read(path)["vertex"].data
    names = data.dtype.names
    xyz = np.stack([data[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    if "f_dc_0" in names:
        color = np.stack([data[f"f_dc_{i}"] for i in range(3)], axis=1) * SH_C0 + 0.5
    else:
        color = np.stack([data[name] for name in ("red", "green", "blue")], axis=1) / 255
    return {
        "means": xyz,
        "colors": color.astype(np.float32),
        "log_scales": np.stack([data[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32),
        "quats": np.stack([data[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32),
        "opacity_logits": np.asarray(data["opacity"], dtype=np.float32),
    }


def write_ply(path, parameters):
    keys = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    count = len(parameters["means"])
    data = np.empty(count, dtype=[(name, "<f4") for name in keys])
    values = np.concatenate((parameters["means"], np.zeros((count, 3)), (parameters["colors"] - 0.5) / SH_C0, parameters["opacity_logits"][:, None], parameters["log_scales"], parameters["quats"]), axis=1)
    for index, name in enumerate(keys):
        data[name] = values[:, index]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def normalize_proxy(parameters, vertices, up="y"):
    rotation = np.eye(3, dtype=np.float32)
    if up == "z":
        rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    transformed = vertices @ rotation.T
    center = (transformed.max(axis=0) + transformed.min(axis=0)) / 2
    scale = 1 / max(np.max(np.ptp(transformed, axis=0)), 1e-8)
    result = {name: value.copy() for name, value in parameters.items()}
    result["means"] = ((parameters["means"] @ rotation.T - center) * scale).astype(np.float32)
    result["log_scales"] += np.log(scale)
    from scipy.spatial.transform import Rotation
    source = Rotation.from_quat(parameters["quats"][:, [1, 2, 3, 0]]).as_matrix()
    result["quats"] = Rotation.from_matrix(rotation @ source).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    return result, ((transformed - center) * scale).astype(np.float32), {"rotation": rotation.tolist(), "center": center.tolist(), "scale": float(scale)}
