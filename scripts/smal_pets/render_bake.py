"""Render full-SMAL front and side poses from the exported skeleton tracks."""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import torch
from pytorch3d.renderer import DirectionalLights, FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, RasterizationSettings, SoftPhongShader, TexturesVertex, look_at_view_transform
from pytorch3d.structures import Meshes
from model import PetModel
from glb import read_glb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bite-source", required=True)
    parser.add_argument("--bite-fit", required=True)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pet = PetModel(args.bite_source, args.bite_fit)
    fit = np.load(args.bite_fit)
    if "coordinate_transform" in fit.files:
        transform = fit["coordinate_transform"]
        scale = np.linalg.norm(transform[:3, 0])
        pet.set_alignment(transform[:3, :3] / scale, scale, transform[:3, 3])
    else:
        pet.set_alignment(fit["alignment"], float(fit["global_scale"]), fit["translation"])
    rest = pet().detach().cpu().numpy()
    alignment = pet.alignment.cpu().numpy()
    document, accessor = read_glb(args.glb)
    rest_position = document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    error = float(np.abs(accessor(rest_position) - rest).max())
    assert error < 2e-5, error
    camera_positions = [(0, 0.30, -2.0), (2.0, 0.30, 0)]
    lights = DirectionalLights(device="cuda", direction=((0.2, -0.5, -1.0),), ambient_color=((0.4, 0.4, 0.4),), diffuse_color=((0.6, 0.6, 0.6),), specular_color=((0, 0, 0),))
    raster_settings = RasterizationSettings(image_size=256, blur_radius=0, faces_per_pixel=1)
    floor_height = float(rest[:, 1].min())
    floor_vertices = torch.as_tensor([[-0.8, floor_height, -0.8], [-0.8, floor_height, 0.8], [0.8, floor_height, 0.8], [0.8, floor_height, -0.8]], dtype=torch.float32, device="cuda")
    floor_faces = torch.as_tensor([[3889, 3890, 3891], [3889, 3891, 3892]], device="cuda")
    faces = torch.cat((pet.faces, floor_faces))[None]
    colors = torch.cat((torch.full((3889, 3), 0.72, device="cuda"), torch.full((4, 3), 0.97, device="cuda")))[None]
    montage = Image.new("RGB", (1024, 6 * 280), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default(size=18)
    samples = {}
    for index, animation in enumerate(document["animations"]):
        name = animation["name"]
        phase = {"sit": 1.0, "walk": 0.25, "run": 0.25, "spin": 0.25, "bark": 5 / 12, "wag": 0.15}.get(name, 0.5)
        rotations = np.repeat(np.eye(3)[None], 35, axis=0)
        root_translation = None
        sampled_time = 0
        for channel in animation["channels"]:
            joint = channel["target"]["node"] - 1
            sampler = animation["samplers"][channel["sampler"]]
            times = accessor(sampler["input"])[:, 0]
            frame = int(np.argmin(np.abs(times - times[-1] * phase)))
            sampled_time = float(times[frame])
            value = accessor(sampler["output"])[frame]
            if channel["target"]["path"] == "rotation":
                rotations[joint] = Rotation.from_quat(value).as_matrix()
            elif joint == 0:
                root_translation = value
        rotations[0] = alignment.T @ rotations[0]
        pose = torch.as_tensor(Rotation.from_matrix(rotations).as_rotvec(), device="cuda", dtype=torch.float32)[None]
        offset = root_translation - np.asarray(document["nodes"][1]["translation"])
        with torch.no_grad():
            vertices = pet(pose, torch.as_tensor(offset, dtype=torch.float32, device="cuda"))
            mesh = Meshes(verts=torch.cat((vertices, floor_vertices))[None], faces=faces, textures=TexturesVertex(verts_features=colors))
            images = []
            for eye in camera_positions:
                center_y = float((vertices[:, 1].min() + vertices[:, 1].max()) / 2)
                target = np.array([0, center_y, 0])
                eye = np.asarray(eye) + [0, center_y, 0]
                rotation, translation = look_at_view_transform(eye=(eye,), at=(target,), up=((0, 1, 0),), device="cuda")
                cameras = FoVPerspectiveCameras(device="cuda", R=rotation, T=translation, fov=40)
                renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftPhongShader(device="cuda", cameras=cameras, lights=lights))
                rgb = renderer(mesh)[0, :, :, :3].clamp(0, 1).cpu().numpy()
                images.append(Image.fromarray((rgb * 255).round().astype(np.uint8)))
        col, row = index % 2, index // 2
        for side, image in enumerate(images):
            image.save(output / f"{name}-{'front' if side == 0 else 'side'}.png")
            montage.paste(image, (col * 512 + side * 256, row * 280 + 24))
        draw.text((col * 512 + 12, row * 280 + 2), f"{name}  {sampled_time:.2f}s", fill="black", font=font)
        samples[name] = sampled_time
    montage.save(output / "motion-montage.png")
    (output / "render-check.json").write_text(json.dumps({"rest_vertex_max_error": error, "samples": samples, "renderer": "full D-SMAL from exported skeleton tracks"}, indent=2) + "\n")
    print(json.dumps({"status": "passed", "rest_vertex_max_error": error, "montage": str(output / "motion-montage.png")}))


if __name__ == "__main__":
    main()
