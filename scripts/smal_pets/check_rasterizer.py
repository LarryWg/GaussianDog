"""Check gsplat CUDA projection, white compositing and backward gradients."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    means = torch.tensor([[0.2, 0.1, 0.0]], device="cuda", requires_grad=True)
    quats = torch.tensor([[0.98, 0.0, 0.2, 0.0]], device="cuda", requires_grad=True)
    scales = torch.tensor([[0.06, 0.04, 0.02]], device="cuda", requires_grad=True)
    colors = torch.tensor([[0.2, 0.4, 0.6]], device="cuda", requires_grad=True)
    opacity = torch.tensor([0.8], device="cuda", requires_grad=True)
    view = torch.eye(4, device="cuda")[None]
    view[0, 2, 3] = 2
    intrinsic = torch.tensor([[[700., 0., 256.], [0., 700., 256.], [0., 0., 1.]]], device="cuda")
    image, alpha, metadata = rasterization(means, quats, scales, opacity, colors, view, intrinsic, 512, 512, packed=False, backgrounds=torch.ones((1, 3), device="cuda"))
    expected = torch.tensor([326., 291.], device="cuda")
    error = float(torch.abs(metadata["means2d"][0, 0] - expected).max())
    assert error < 1e-4, error
    assert torch.allclose(image[0, 0, 0], torch.ones(3, device="cuda"), atol=1e-7)
    loss = image.square().mean()
    loss.backward()
    gradients = {name: bool(value.grad is not None and torch.isfinite(value.grad).all()) for name, value in (("means", means), ("quaternions", quats), ("scales", scales), ("colors", colors), ("opacity", opacity))}
    assert all(gradients.values()), gradients
    pixels = image[0].detach().clamp(0, 1).cpu().numpy()
    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(output / "rasterizer-check.png")
    report = {"status": "passed", "device": torch.cuda.get_device_name(), "torch": torch.__version__, "projection_max_error_pixels": error, "gradients_finite": gradients, "alpha_max": float(alpha.max())}
    (output / "rasterizer-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
