"""Check upstream vertex parity and differentiability of the complete model."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import PetModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bite-source", required=True)
    parser.add_argument("--bite-fit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pet = PetModel(args.bite_source, args.bite_fit)
    fit = np.load(args.bite_fit)
    vertices = pet()
    assert tuple(vertices.shape) == (3889, 3)
    assert tuple(pet.smal.weights.shape) == (3889, 35)
    assert pet.betas.numel() == 30 and pet.limbs.numel() == 7
    assert pet.smal.posedirs.shape[0] == 306
    key = "native_verts" if "native_verts" in fit.files else "verts"
    expected = fit[key].reshape(3889, 3)
    if key == "verts" and "trans" in fit.files:
        expected = expected - fit["trans"].reshape(1, 3)
    error = float(np.abs(vertices.detach().cpu().numpy() - expected).max())
    assert error < 2e-5, error
    regularizers = pet.regularization(vertices)
    expected_offsets = float(torch.linalg.vector_norm(pet.offsets.detach()).square())
    assert np.isclose(float(regularizers["offsets"].detach()), expected_offsets)
    loss = vertices.square().mean() + sum(regularizers.values())
    loss.backward()
    gradients = {name: bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()) for name, parameter in (("betas", pet.betas), ("limbs", pet.limbs), ("pose", pet.pose), ("offsets", pet.offsets))}
    assert all(gradients.values()), gradients
    sorted_weights = torch.sort(pet.smal.weights, descending=True, dim=1).values
    report = {"status": "passed", "native_vertex_max_error": error, "gradients_finite": gradients, "full_joint_count": 35, "pose_corrective_basis": list(pet.smal.posedirs.shape), "pose_corrective_norm": float(pet.smal.posedirs.norm()), "offset_squared_l2": expected_offsets, "geometric_reduction": "mean", "parameter_prior_reduction": "sum", "maximum_skin_mass_beyond_four": float(sorted_weights[:, 4:].sum(-1).max())}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
