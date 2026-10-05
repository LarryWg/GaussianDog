"""Generate the standing appearance proxy with official TRELLIS weights."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from huggingface_hub import snapshot_download
    from PIL import Image
    import torch
    import trimesh
    from trellis.pipelines import TrellisImageTo3DPipeline

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
    weights = snapshot_download("microsoft/TRELLIS-image-large", revision=revision)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(weights)
    pipeline.cuda()
    image = Image.open(args.image).convert("RGB")
    image = pipeline.preprocess_image(image)
    image.save(output / "processed-reference.png")
    with torch.inference_mode():
        assets = pipeline.run(image, seed=42, formats=["gaussian", "mesh"], preprocess_image=False)
    gaussian = assets["gaussian"][0]
    gaussian.save_ply(str(output / "proxy.ply"), transform=None)
    mesh = assets["mesh"][0]
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    if not mesh.success or not len(faces):
        raise RuntimeError("TRELLIS did not produce a usable structural mesh")
    structural_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    structural_mesh.export(output / "proxy.obj")
    if mesh.vertex_attrs is not None:
        colors = mesh.vertex_attrs.detach().cpu().numpy()
        structural_mesh.visual.vertex_colors = (colors[:, :3].clip(0, 1) * 255).astype("uint8")
        structural_mesh.export(output / "proxy.glb")
    (output / "provenance.json").write_text(json.dumps({
        "model": "microsoft/TRELLIS-image-large",
        "revision": revision,
        "seed": 42,
        "image": str(args.image),
        "vertices": len(vertices),
        "faces": len(faces),
        "appearance": "TRELLIS Gaussian RGB",
        "coordinateSpace": "TRELLIS z-up",
    }, indent=2) + "\n")
    print("TRELLIS_PROXY_COMPLETE", len(vertices), len(faces), flush=True)


if __name__ == "__main__":
    main()
