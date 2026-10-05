"""Render the fitted triangle mesh beside its calibrated proxy targets."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from pytorch3d.renderer import DirectionalLights, MeshRasterizer, MeshRenderer, RasterizationSettings, SoftPhongShader, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


def render_structure(vertices, faces, view, resolution):
    mesh = Meshes(verts=vertices[None], faces=faces[None], textures=TexturesVertex(verts_features=torch.full_like(vertices[None], 0.72)))
    lights = DirectionalLights(device="cuda", direction=((0.2, -0.5, -1),), ambient_color=((0.4, 0.4, 0.4),), diffuse_color=((0.6, 0.6, 0.6),), specular_color=((0, 0, 0),))
    settings = RasterizationSettings(image_size=resolution, blur_radius=0, faces_per_pixel=1)
    extrinsic = torch.as_tensor(view["view"], dtype=torch.float32, device="cuda")
    cameras = cameras_from_opencv_projection(R=extrinsic[:3, :3][None], tvec=extrinsic[:3, 3][None], camera_matrix=torch.as_tensor(view["K"], dtype=torch.float32, device="cuda")[None], image_size=torch.as_tensor([[resolution, resolution]], device="cuda"))
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=settings), shader=SoftPhongShader(device="cuda", cameras=cameras, lights=lights))
    return renderer(mesh)[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    args = parser.parse_args()
    directory = Path(args.training)
    fit = np.load(directory / "initial_alignment.npz")
    configuration = json.loads((directory / "cameras.json").read_text())
    selected = json.loads((directory / "target-montage.json").read_text())
    resolution = configuration["resolution"]
    vertices = torch.as_tensor(fit["vertices"], dtype=torch.float32, device="cuda")
    faces = torch.as_tensor(fit["faces"], dtype=torch.long, device="cuda")
    montage = Image.new("RGB", (resolution * 3, (resolution + 24) * 2), "white")
    draw = ImageDraw.Draw(montage)
    with torch.no_grad():
        for row, name in enumerate(("front", "side")):
            view = configuration["views"][selected[name]]
            rendered = render_structure(vertices, faces, view, resolution)
            target = np.asarray(Image.open(directory / view["file"]).convert("RGB")) / 255
            gray = rendered[:, :, :3].clip(0, 1)
            overlay = target.copy()
            mask = rendered[:, :, 3] > 0
            overlay[mask] = 0.5 * target[mask] + 0.5 * gray[mask]
            for column, (label, pixels) in enumerate((("proxy", target), ("structure", gray), ("overlay", overlay))):
                position = (column * resolution, row * (resolution + 24))
                draw.text((position[0] + 8, position[1] + 6), f"{name} {label}", fill="black")
                montage.paste(Image.fromarray((pixels * 255).round().astype(np.uint8)), (position[0], position[1] + 24))
    path = directory / "initial_mesh_alignment.png"
    montage.save(path)
    print(json.dumps({"status": "rendered", "path": str(path), "calibratedViews": {name: selected[name] for name in ("front", "side")}}))


if __name__ == "__main__":
    main()
