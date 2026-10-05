"""Check PLY conventions and deterministic spatial density prefixes."""

import json
import tempfile
from pathlib import Path
import numpy as np
from gaussians import normalize_proxy, progressive_order, read_ply, write_ply


def main():
    rng = np.random.default_rng(42)
    count = 1500
    clusters = np.repeat(np.arange(3), [1000, 400, 100])
    points = rng.normal(size=(count, 3)).astype(np.float32) * 1e-5
    points[:, 0] += clusters
    parameters = {"means": points, "colors": rng.uniform(0, 1, (count, 3)).astype(np.float32), "log_scales": rng.uniform(-9, -4, (count, 3)).astype(np.float32), "opacity_logits": rng.uniform(-4, 6, count).astype(np.float32), "quats": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 1))}
    order = progressive_order(parameters)
    assert np.array_equal(order, progressive_order(parameters))
    assert len(np.unique(order)) == count
    assert len(np.unique(clusters[order[:32]])) == 3
    sorted_parameters = {name: value[order] for name, value in parameters.items()}
    normalized, vertices, _ = normalize_proxy(parameters, points.astype(np.float64), "z")
    assert all(value.dtype == np.float32 for value in normalized.values())
    assert vertices.dtype == np.float32
    assert np.isclose(np.ptp(vertices, axis=0).max(), 1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.ply"
        write_ply(path, sorted_parameters)
        restored = read_ply(path)
        errors = {name: float(np.abs(restored[name] - value).max()) for name, value in sorted_parameters.items()}
        assert all(error < 2e-6 for error in errors.values()), errors
        for prefix in (500, 1000, 1500):
            write_ply(path, {name: value[:prefix] for name, value in sorted_parameters.items()})
            assert np.array_equal(read_ply(path)["means"], restored["means"][:prefix])
    print(json.dumps({"status": "passed", "roundtrip_errors": errors, "checks": ["spatial_coverage", "determinism", "all_rows_retained", "identical_prefixes", "float64_mesh_to_float32_gaussians"]}))


if __name__ == "__main__":
    main()
