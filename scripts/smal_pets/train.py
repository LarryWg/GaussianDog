"""Direct SMAL-pets bound and free Gaussian optimization."""

import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from gsplat import rasterization
import trimesh

from gaussians import normalize_proxy, read_ply, write_ply
from geometry import align_mesh, body_forward, camera_views
from model import PetModel


def ssim(a, b):
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    x = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    gaussian = torch.exp(-x.square() / 4.5)
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None].expand(3, 1, 11, 11)
    conv = lambda value: F.conv2d(value, kernel, padding=5, groups=3)
    ma, mb = conv(a), conv(b)
    va = conv(a * a) - ma.square()
    vb = conv(b * b) - mb.square()
    cov = conv(a * b) - ma * mb
    return (((2 * ma * mb + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / ((ma.square() + mb.square() + 0.01 ** 2) * (va + vb + 0.03 ** 2))).mean()


def render(parameters, view, resolution):
    image, alpha, info = rasterization(means=parameters["means"], quats=F.normalize(parameters["quats"], dim=-1), scales=parameters["log_scales"].exp(), opacities=parameters["opacity_logits"].sigmoid(), colors=parameters["colors"], viewmats=torch.as_tensor(view["view"], dtype=torch.float32, device="cuda")[None], Ks=torch.as_tensor(view["K"], dtype=torch.float32, device="cuda")[None], width=resolution, height=resolution, backgrounds=torch.ones((1, 3), device="cuda"), packed=False)
    return image[0], alpha[0], info


class BoundGaussians(nn.Module):
    def __init__(self, pet, proxy, per_face=4):
        super().__init__()
        self.register_buffer("pet_faces", pet.faces)
        self.register_buffer("face_ids", torch.arange(len(pet.faces), device="cuda").repeat_interleave(per_face))
        bary = torch.rand((len(self.face_ids), 3), device="cuda") + 0.2
        self.barycentric = nn.Parameter(bary.log())
        with torch.no_grad():
            points = self.geometry(pet())["means"].cpu().numpy()
        _, closest = cKDTree(proxy["means"]).query(points)
        self.colors = nn.Parameter(torch.as_tensor(proxy["colors"][closest], device="cuda"))
        self.opacity_logits = nn.Parameter(torch.full((len(points),), math.log(0.1 / 0.9), device="cuda"))

    def geometry(self, vertices):
        triangles = vertices[self.pet_faces[self.face_ids]]
        center = triangles.mean(dim=1)
        normal = F.normalize(torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1)
        t1 = triangles[:, 1] - center
        e1 = F.normalize(t1, dim=-1)
        t2 = triangles[:, 2] - center
        t2p = t2 - (t2 * normal).sum(-1, keepdim=True) * normal - (t2 * e1).sum(-1, keepdim=True) * e1
        e2 = F.normalize(t2p, dim=-1)
        rotation = torch.stack((normal, e1, e2), dim=-1)
        scales = torch.stack((torch.full_like(t1[:, 0], 1e-4), torch.linalg.vector_norm(t1, dim=-1) * 0.5, (t2 * e2).sum(-1).abs() * 0.5), dim=-1).clamp_min(1e-6)
        return {"means": (triangles * self.barycentric.softmax(-1)[:, :, None]).sum(1), "quats": matrix_to_quaternion(rotation), "log_scales": scales.log()}

    def parameters_at(self, vertices):
        return {**self.geometry(vertices), "colors": self.colors, "opacity_logits": self.opacity_logits}

    def optimizer_groups(self):
        return [{"params": [self.barycentric], "lr": 0.01, "name": "barycentric"}, {"params": [self.colors], "lr": 0.0025, "name": "colors"}, {"params": [self.opacity_logits], "lr": 0.025, "name": "opacity"}]


class FreeGaussians(nn.ParameterDict):
    def __init__(self, parameters):
        super().__init__({name: nn.Parameter(value.detach().clone()) for name, value in parameters.items()})

    def optimizer_groups(self):
        rates = {"means": 0.00016, "colors": 0.0025, "log_scales": 0.005, "quats": 0.001, "opacity_logits": 0.025}
        return [{"params": [self[name]], "lr": rate, "name": name} for name, rate in rates.items()]

    def resize(self, optimizer, keep, extra, split):
        extra_scales = self["log_scales"].detach()[extra].exp()
        extra_quats = self["quats"].detach()[extra]
        for name, old in list(self.items()):
            values = torch.cat((old.detach()[keep], old.detach()[extra]), dim=0)
            if name == "means" and len(extra):
                noise = torch.randn_like(values[len(keep):]) * extra_scales * split[:, None]
                from pytorch3d.transforms import quaternion_apply
                values[len(keep):] += quaternion_apply(F.normalize(extra_quats, dim=-1), noise)
            if name == "log_scales" and len(extra):
                values[len(keep):] -= split[:, None].float() * math.log(1.6)
            replacement = nn.Parameter(values)
            for group in optimizer.param_groups:
                if group["name"] != name:
                    continue
                state = optimizer.state.pop(old, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key] = torch.cat((state[key][keep], torch.zeros_like(old[extra])), dim=0)
                optimizer.state[replacement] = state
                group["params"] = [replacement]
            self[name] = replacement


def numpy_parameters(parameters):
    return {name: value.detach().cpu().numpy() for name, value in parameters.items()}


def density_indices(opacity, score, scales, cap):
    keep = torch.where(opacity >= 0.005)[0]
    if not len(keep):
        raise RuntimeError("Densification pruned every Gaussian")
    room = max(0, cap - len(keep))
    extra = keep[score[keep] > 2e-4]
    if len(extra) > room:
        extra = extra[torch.topk(score[extra], room).indices]
    split_ids = extra[scales[extra].amax(-1) > 0.01]
    clone_ids = extra[scales[extra].amax(-1) <= 0.01]
    retained = keep[~torch.isin(keep, split_ids)]
    children = torch.cat((clone_ids, split_ids.repeat_interleave(2)))
    split = torch.cat((torch.zeros(len(clone_ids), dtype=torch.bool, device=opacity.device), torch.ones(2 * len(split_ids), dtype=torch.bool, device=opacity.device)))
    return retained, children, split


def prepare_targets(args):
    output = Path(args.output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.proxy_mesh, force="mesh", process=False)
    proxy, vertices, transform = normalize_proxy(read_ply(args.proxy_ply), np.asarray(mesh.vertices), args.proxy_up)
    write_ply(output / "proxy_normalized.ply", proxy)
    np.savez(output / "proxy_mesh.npz", vertices=vertices.astype(np.float32), faces=np.asarray(mesh.faces, dtype=np.int32))
    views = camera_views(args.views, args.resolution)
    tensor_proxy = {name: torch.as_tensor(value, dtype=torch.float32, device="cuda") for name, value in proxy.items()}
    with torch.no_grad():
        for view in views:
            image, alpha, _ = render(tensor_proxy, view, args.resolution)
            pixels = torch.cat((image, alpha), dim=-1).clamp(0, 1).cpu().numpy()
            Image.fromarray((pixels * 255).round().astype(np.uint8), "RGBA").save(output / view["file"])
    configuration = {"resolution": args.resolution, "views": views, "proxy_transform": transform, "target_source": "trellis-gaussians", "seed": 42, "coordinateSpace": "mesh-local-y-up"}
    (output / "cameras.json").write_text(json.dumps(configuration, indent=2) + "\n")
    return proxy, vertices, configuration


def write_target_montage(configuration, output, forward):
    resolution = configuration["resolution"]
    views = configuration["views"]
    up = np.array([0, 1, 0])
    side = np.cross(forward, up)
    montage = Image.new("RGB", (2 * resolution, 2 * (resolution + 24)), "white")
    draw = ImageDraw.Draw(montage)
    directions = {"front": forward + 0.2 * up, "side": side + 0.2 * up, "rear": -forward + 0.2 * up, "three quarter": forward + 0.7 * side + 0.2 * up}
    selected = {}
    for index, (label, direction) in enumerate(directions.items()):
        direction = np.asarray(direction) / np.linalg.norm(direction)
        view = max(views, key=lambda item: np.dot(np.linalg.inv(item["view"])[:3, 3] / 2, direction))
        col, row = index % 2, index // 2
        montage.paste(Image.open(output / view["file"]).convert("RGB"), (col * resolution, row * (resolution + 24) + 24))
        draw.text((col * resolution + 8, row * (resolution + 24) + 6), f"{label}  calibrated view {view['id']}", fill="black")
        selected[label] = view["id"]
    montage.save(output / "target-montage.png")
    (output / "target-montage.json").write_text(json.dumps(selected, indent=2) + "\n")


def validate(parameters, configuration, output, step):
    metrics = []
    with torch.no_grad():
        for view in configuration["views"]:
            if view["split"] != "holdout":
                continue
            predicted, _, _ = render(parameters, view, configuration["resolution"])
            target = torch.as_tensor(np.asarray(Image.open(output / view["file"]).convert("RGB")).copy(), device="cuda") / 255
            mse = F.mse_loss(predicted, target).clamp_min(1e-10)
            metrics.append({"view": view["id"], "psnr": float(-10 * mse.log10()), "ssim": float(ssim(predicted, target))})
            if view["id"] == 0:
                comparison = np.concatenate((target.cpu().numpy(), predicted.clamp(0, 1).cpu().numpy()), axis=1)
                Image.fromarray((comparison * 255).round().astype(np.uint8)).save(output / f"validation-{step:05d}.png")
    return {"step": step, "psnr": float(np.mean([item["psnr"] for item in metrics])), "ssim": float(np.mean([item["ssim"] for item in metrics])), "views": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bite-source", required=True)
    parser.add_argument("--bite-fit", required=True)
    parser.add_argument("--proxy-ply", required=True)
    parser.add_argument("--proxy-mesh", required=True)
    parser.add_argument("--proxy-up", choices=("y", "z"), default="z")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bound-steps", type=int, default=15000)
    parser.add_argument("--free-steps", type=int, default=25000)
    parser.add_argument("--views", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cap", type=int, default=150000)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--initialization", help="JSON similarity transform and optional joint and limb settings")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "cameras.json").exists():
        configuration = json.loads((output / "cameras.json").read_text())
        proxy = read_ply(output / "proxy_normalized.ply")
        vertices = np.load(output / "proxy_mesh.npz")["vertices"]
    else:
        proxy, vertices, configuration = prepare_targets(args)
    args.resolution = configuration["resolution"]
    pet = PetModel(args.bite_source, args.bite_fit)
    initialization = None
    if args.initialization:
        initialization = json.loads(Path(args.initialization).read_text())
        rotation = np.asarray(initialization["rotation"], dtype=np.float64)
        scale = float(initialization["scale"])
        translation = np.asarray(initialization["translation"], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or scale <= 0 or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Invalid anatomical similarity transform")
        with torch.no_grad():
            for joint, vector in initialization.get("joint_rotvec", {}).items():
                pet.pose[0, int(joint)] = torch.as_tensor(vector, dtype=torch.float32, device="cuda")
            for index, value in initialization.get("limb_values", {}).items():
                pet.limbs[0, int(index)] = float(value)
            pet.initial_pose.copy_(pet.pose.detach())
        (output / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")
    else:
        rotation, scale, translation = align_mesh(pet().detach().cpu().numpy(), vertices)
    pet.set_alignment(rotation, scale, translation)
    bound = BoundGaussians(pet, proxy)
    aligned = pet().detach().cpu().numpy()
    forward = body_forward(aligned)
    write_target_montage(configuration, output, forward)
    (output / "alignment.json").write_text(json.dumps({"native_up_to_world": (rotation @ np.array([0, 0, 1])).tolist(), "body_forward": forward.tolist(), "nose_forward": (aligned[1863] - aligned[452]).tolist(), "scale": scale, "translation": translation.tolist(), "rotation": rotation.tolist(), "fit": "anatomical anchors" if initialization else "yaw-only upright ICP"}, indent=2) + "\n")
    pet.export_fit(output / "initial_alignment.npz")
    trimesh.Trimesh(vertices=pet().detach().cpu().numpy(), faces=pet.faces.cpu().numpy(), process=False).export(output / "initial_alignment.obj")
    preview_views = []
    for direction in (forward + np.array([0, 0.2, 0]), np.cross(forward, [0, 1, 0]) + np.array([0, 0.2, 0])):
        direction = np.asarray(direction) / np.linalg.norm(direction)
        preview_views.append(max(configuration["views"], key=lambda view: np.dot(np.linalg.inv(view["view"])[:3, 3] / 2, direction)))
    previews = []
    with torch.no_grad():
        preview_parameters = bound.parameters_at(pet())
        preview_parameters["opacity_logits"] = torch.full_like(preview_parameters["opacity_logits"], 8)
        for view in preview_views:
            image, _, _ = render(preview_parameters, view, args.resolution)
            target = np.asarray(Image.open(output / view["file"]).convert("RGB")) / 255
            previews.append(np.concatenate((target, image.clamp(0, 1).cpu().numpy()), axis=1))
    Image.fromarray((np.concatenate(previews, axis=0) * 255).round().astype(np.uint8)).save(output / "initial_alignment.png")
    free = None
    stage = "bound"
    step = 0
    history = []
    optimizer = torch.optim.Adam(pet.optimizer_groups() + bound.optimizer_groups(), betas=(0.9, 0.999), eps=1e-8)
    gradient_sum = None
    gradient_count = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        pet.load_state_dict(saved["pet"])
        stage, step, history = saved["stage"], saved["step"], saved["history"]
        if stage == "bound":
            bound.load_state_dict(saved["gaussians"])
        else:
            free = FreeGaussians(saved["gaussians"])
            optimizer = torch.optim.Adam(pet.optimizer_groups(0.1) + free.optimizer_groups(), betas=(0.9, 0.999), eps=1e-8)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        random.setstate(saved["python_rng"])
        if "numpy_rng" in saved:
            np.random.set_state(saved["numpy_rng"])
        gradient_sum = saved.get("gradient_sum")
        gradient_count = saved.get("gradient_count")
    train_views = [view for view in configuration["views"] if view["split"] == "train"]
    targets = {view["id"]: torch.as_tensor(np.asarray(Image.open(output / view["file"]).convert("RGB")).copy(), device="cuda", dtype=torch.float32) / 255 for view in train_views}
    started = time.monotonic()
    stage_started = started
    stage_completed = 0
    completed_now = 0
    checked_stages = set()

    def checkpoint(filename="checkpoint.pt"):
        parameters = bound.parameters_at(pet()) if stage == "bound" else free
        checkpoint_path = output / filename
        temporary_path = output / f".{filename}.tmp"
        torch.save({"pet": pet.state_dict(), "gaussians": bound.state_dict() if stage == "bound" else free.state_dict(), "optimizer": optimizer.state_dict(), "stage": stage, "step": step, "history": history, "args": vars(args), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "gradient_sum": gradient_sum, "gradient_count": gradient_count}, temporary_path)
        temporary_path.replace(checkpoint_path)
        write_ply(output / "latest.ply", numpy_parameters(parameters))
        pet.export_fit(output / "optimized_fit.npz")
        (output / "training.json").write_text(json.dumps({"stage": stage, "step": step, "bound_steps": args.bound_steps, "free_steps": args.free_steps, "complete": stage == "free" and step >= args.free_steps, "count": len(parameters["means"]), "elapsed_seconds_this_run": time.monotonic() - started, "configuration": configuration, "defaults": vars(args), "validation": history}, indent=2) + "\n")

    try:
        while True:
            if stage == "bound" and step >= args.bound_steps:
                parameters = bound.parameters_at(pet())
                write_ply(output / "bound.ply", numpy_parameters(parameters))
                free = FreeGaussians(parameters)
                stage, step = "free", 0
                optimizer = torch.optim.Adam(pet.optimizer_groups(0.1) + free.optimizer_groups(), betas=(0.9, 0.999), eps=1e-8)
                checkpoint("bound-checkpoint.pt")
                stage_started = time.monotonic()
                stage_completed = 0
            if stage == "free" and step >= args.free_steps:
                break
            if args.stop_after is not None and completed_now >= args.stop_after:
                break
            optimizer.zero_grad(set_to_none=True)
            mesh_vertices = pet()
            parameters = bound.parameters_at(mesh_vertices) if stage == "bound" else free
            view = random.choice(train_views)
            image, _, info = render(parameters, view, args.resolution)
            if stage == "free":
                info["means2d"].retain_grad()
                for group in optimizer.param_groups:
                    if group["name"] == "means":
                        group["lr"] = math.exp(math.log(0.00016) * (1 - step / args.free_steps) + math.log(0.0000016) * step / args.free_steps)
            rgb = 0.8 * F.l1_loss(image, targets[view["id"]]) + 0.2 * (1 - ssim(image, targets[view["id"]]))
            regularizers = pet.regularization(mesh_vertices)
            loss = rgb + sum(regularizers.values())
            opacity = parameters["opacity_logits"].sigmoid()
            if stage == "bound":
                loss = loss - 0.001 * opacity.mean()
            else:
                distances = _PointFaceDistance.apply(parameters["means"].detach(), torch.zeros(1, dtype=torch.long, device="cuda"), mesh_vertices[pet.faces], torch.zeros(1, dtype=torch.long, device="cuda"), len(parameters["means"]), 1e-8)
                scales = parameters["log_scales"].exp()
                loss = loss + 10 * distances.clamp_min(1e-12).sqrt().mean() + (scales.square().sum(-1) + scales.max(-1).values - scales.min(-1).values).mean() - 0.001 * (opacity * opacity.clamp_min(1e-8).log()).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {stage} loss at step {step}")
            loss.backward()
            if stage not in checked_stages:
                gradients = [value.grad for group in optimizer.param_groups for value in group["params"] if value.grad is not None]
                if not gradients or not torch.stack([torch.isfinite(value).all() for value in gradients]).all():
                    raise RuntimeError(f"Invalid {stage} parameter gradients")
                checked_stages.add(stage)
            optimizer.step()
            with torch.no_grad():
                parameters["colors"].clamp_(0, 1)
                parameters["opacity_logits"].clamp_(-12, 12)
                if stage == "free":
                    free["quats"].copy_(F.normalize(free["quats"], dim=-1))
                    free["log_scales"].clamp_(math.log(1e-5), math.log(0.1))
                    projected_gradient = info["means2d"].grad
                    visible = info["radii"].amax(-1)[0] > 0
                    gradient = torch.linalg.vector_norm(projected_gradient[0] * args.resolution / 2, dim=-1)
                    if gradient_sum is None or len(gradient_sum) != len(gradient):
                        gradient_sum = torch.zeros_like(gradient)
                        gradient_count = torch.zeros_like(gradient)
                    gradient_sum += gradient * visible
                    gradient_count += visible
                    if (step + 1) % 100 == 0 and step < 15000:
                        score = gradient_sum / gradient_count.clamp_min(1)
                        retained, children, split = density_indices(free["opacity_logits"].sigmoid(), score, free["log_scales"].exp(), args.cap)
                        free.resize(optimizer, retained, children, split)
                        gradient_sum = gradient_count = None
                    if (step + 1) % 3000 == 0 and step < 15000:
                        free["opacity_logits"].clamp_(max=math.log(0.01 / 0.99))
                        opacity_state = optimizer.state.get(free["opacity_logits"], {})
                        for moment in ("exp_avg", "exp_avg_sq"):
                            if moment in opacity_state:
                                opacity_state[moment].zero_()
            step += 1
            completed_now += 1
            stage_completed += 1
            if step % 100 == 0:
                elapsed = time.monotonic() - started
                print(json.dumps({"stage": stage, "step": step, "loss": float(loss.detach()), "rgb": float(rgb.detach()), "regularizers": {name: float(value.detach()) for name, value in regularizers.items()}, "count": len(bound.face_ids) if stage == "bound" else len(free["means"]), "steps_per_second": completed_now / elapsed, "stage_steps_per_second": stage_completed / (time.monotonic() - stage_started)}), flush=True)
            if step % 1000 == 0:
                current = bound.parameters_at(pet()) if stage == "bound" else free
                history.append(validate(current, configuration, output, step + (args.bound_steps if stage == "free" else 0)))
                checkpoint()
    finally:
        checkpoint()
    current = bound.parameters_at(pet()) if stage == "bound" else free
    history.append(validate(current, configuration, output, step + (args.bound_steps if stage == "free" else 0)))
    checkpoint()


if __name__ == "__main__":
    main()
