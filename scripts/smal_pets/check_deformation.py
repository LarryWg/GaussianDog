"""Make a deterministic NumPy reference fixture for the browser kernel."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from geometry import bind_faces, deform


def make_fixture(path):
    rng = np.random.default_rng(42)
    centers = rng.uniform(-0.5, 0.5, (16, 3))
    triangles = centers[:, None] + np.array([[0, 0, 0], [0.10, 0, 0], [0.01, 0.09, 0]])[None]
    rest = triangles.reshape(-1, 3)
    faces = np.arange(len(rest), dtype=np.uint32).reshape(-1, 3)
    points = rng.uniform(-0.6, 0.6, (64, 3))
    rotations = Rotation.random(64, random_state=rng)
    quaternions = rotations.as_quat()[:, [3, 0, 1, 2]]
    scales = rng.uniform(0.001, 0.02, (64, 3))
    indices, weights = bind_faces(points, rest, faces)
    rigid_rotation = Rotation.from_rotvec([0.2, -0.4, 0.1]).as_matrix()
    translation = np.array([0.1, -0.2, 0.3])
    deformations = {"identity": rest.copy(), "rigid": rest @ rigid_rotation.T + translation, "stretch": rest * 1.44}
    bend = rest.copy().reshape(-1, 3, 3)
    for index in range(len(bend)):
        delta = Rotation.from_rotvec([0, centers[index, 2] * 1.8, centers[index, 0] * 0.3]).as_matrix()
        bend[index] = (bend[index] - centers[index]) @ delta.T + centers[index]
    deformations["articulated"] = bend.reshape(-1, 3)
    degenerate = rest.copy()
    degenerate[faces[0]] = degenerate[faces[0, 0]]
    deformations["degenerate"] = degenerate
    cases = []
    results = {}
    for name, posed in deformations.items():
        output, output_q, output_s = deform(points, quaternions, scales, rest, posed, faces, indices, weights)
        assert np.all(np.isfinite(output))
        assert np.allclose(np.linalg.norm(output_q, axis=1), 1, atol=1e-7)
        cases.append({"name": name, "posedPositions": posed.reshape(-1).tolist(), "positions": output.reshape(-1).tolist(), "quaternions": output_q[:, [1, 2, 3, 0]].reshape(-1).tolist(), "scales": output_s.reshape(-1).tolist()})
        results[name] = (output, output_q, output_s)
    assert np.allclose(results["identity"][0], points, atol=1e-7)
    assert np.allclose(results["identity"][2], scales, atol=1e-8)
    assert np.allclose(results["rigid"][0], points @ rigid_rotation.T + translation, atol=1e-7)
    assert np.allclose(results["rigid"][2], scales, atol=1e-8)
    assert np.allclose(results["stretch"][2], scales * 1.2, atol=1e-8)
    frame_rotation = Rotation.from_rotvec([0, 1.2, 0])
    frame_matrix = frame_rotation.as_matrix()
    frame_quaternions = (frame_rotation * rotations).as_quat()[:, [3, 0, 1, 2]]
    framed = deform(points @ frame_matrix.T, frame_quaternions, scales, rest @ frame_matrix.T, deformations["articulated"] @ frame_matrix.T, faces, indices, weights)
    expected_quaternions = (frame_rotation * Rotation.from_quat(results["articulated"][1][:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    assert np.allclose(framed[0], results["articulated"][0] @ frame_matrix.T, atol=1e-7)
    assert np.allclose(np.abs((framed[1] * expected_quaternions).sum(1)), 1, atol=1e-7)
    assert np.allclose(framed[2], results["articulated"][2], atol=1e-8)
    fixture = {"version": 1, "nearestFaces": 10, "count": len(points), "vertexCount": len(rest), "faceCount": len(faces), "restPositions": rest.reshape(-1).tolist(), "faces": faces.reshape(-1).tolist(), "positions": points.reshape(-1).tolist(), "quaternions": quaternions[:, [1, 2, 3, 0]].reshape(-1).tolist(), "scales": scales.reshape(-1).tolist(), "faceIds": indices.reshape(-1).tolist(), "weights": weights.reshape(-1).tolist(), "cases": cases, "tolerance": 2e-6}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps({"status": "passed", "identity_max_error": float(np.abs(results["identity"][0] - points).max()), "rigid_max_error": float(np.abs(results["rigid"][0] - (points @ rigid_rotation.T + translation)).max()), "stretch_max_error": float(np.abs(results["stretch"][2] - scales * 1.2).max()), "frame_rotation_max_error": float(np.abs(framed[0] - results["articulated"][0] @ frame_matrix.T).max()), "fixture": str(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    make_fixture(parser.parse_args().output)
