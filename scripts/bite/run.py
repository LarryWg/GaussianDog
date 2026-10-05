"""Fit one photograph with the published BITE inference and optimization."""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from compat import patch_source


CHECKPOINT = "cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0_forrelease_v0/checkpoint.pth.tar"
REVISION = "4b24155a5bf43ea09b2280386e69783f7af6fab8"


def full_offsets(smal, compact):
    zero = torch.zeros((compact.shape[0], smal.n_center), device=compact.device)
    center = torch.stack((compact[:, :smal.n_center], zero, compact[:, smal.n_center:smal.sl]), dim=1)
    left = torch.stack(tuple(compact[:, smal.sl + k * smal.n_left:smal.sl + (k + 1) * smal.n_left] for k in range(3)), dim=1)
    right = left * torch.tensor([1, -1, 1], device=compact.device)[None, :, None]
    half = torch.cat((center, left, right), dim=2)
    return half.index_select(2, smal.inds_back_torch.to(compact.device)).permute(0, 2, 1)


def finite_tensor(name, value):
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bite-source", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=301)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 301:
        parser.error("--steps must be between 1 and 301")
    source = args.bite_source.resolve()
    output = args.output.resolve()
    args.image = args.image.resolve()
    output.mkdir(parents=True, exist_ok=True)
    patch_source(source)
    os.chdir(source)
    sys.path.insert(0, str(source / "src"))
    if not torch.cuda.is_available():
        raise RuntimeError("BITE requires the authorized CUDA pod")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)
    device = torch.device("cuda")

    from pytorch3d.structures import Meshes
    from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
    from pytorch3d.renderer import FoVPerspectiveCameras, HardPhongShader, MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings, TexturesVertex, look_at_view_transform
    from configs.barc_cfg_defaults import update_cfg_global_with_yaml, get_cfg_global_updated
    from configs.SMAL_configs import SMAL_MODEL_CONFIG
    from stacked_hourglass.datasets.imgcropslist import ImgCrops
    from stacked_hourglass.datasets.utils_dataset_selection import get_norm_dict
    from test_time_optimization.bite_inference_model_for_ttopt import BITEInferenceModel
    from test_time_optimization.utils.utils_ttopt import get_optimed_pose_with_glob
    from lifting_to_3d.utils.geometry_utils import rotmat_to_rot6d
    from smal_pytorch.smal_model.smal_torch_new import SMAL
    from smal_pytorch.renderer.differentiable_renderer import SilhRenderer
    from combined_model.loss_utils.loss_utils import leg_sideway_error, leg_torsion_error, tail_sideway_error, tail_torsion_error, spine_torsion_error, spine_sideway_error
    from combined_model.loss_utils.loss_utils_gc import calculate_plane_errors_batch
    from combined_model.loss_utils.loss_laplacian_mesh_comparison import LaplacianCTF

    update_cfg_global_with_yaml(str(source / "src/configs/refinement_cfg_test_withvertexwisegc_csaddnonflat.yaml"))
    cfg = get_cfg_global_updated()
    image = np.array(Image.open(args.image).convert("RGB"))
    dataset = ImgCrops(image_list=[image], bbox_list=None, dataset_mode="complete")
    data_info = dataset.DATA_INFO
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    input_image, _ = next(iter(loader))
    input_image = input_image.float().to(device)
    model = BITEInferenceModel(cfg, str(source / "checkpoint" / CHECKPOINT), get_norm_dict(data_info, device))
    gates = {"checkpoint_loaded": True, "source_revision": REVISION, "steps_requested": args.steps}
    renderer = SilhRenderer(image_size=256).to(device)
    triangle = torch.tensor([[[-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.0, 0.2, 1.0]]], device=device, requires_grad=True)
    triangle_faces = torch.tensor([[[0, 1, 2]]], device=device)
    smoke_silhouette, _ = renderer(vertices=triangle, points=triangle, faces=triangle_faces, focal_lengths=torch.tensor([[150.0]], device=device))
    finite_tensor("CUDA silhouette", smoke_silhouette)
    if smoke_silhouette.max() < 0.5:
        raise RuntimeError("CUDA rasterizer did not render the smoke triangle")
    smoke_silhouette.mean().backward()
    finite_tensor("CUDA rasterizer gradient", triangle.grad)
    gates["cuda_rasterization"] = True
    with torch.no_grad():
        predictions = model.get_all_results(input_image)
        result = model.get_selected_results(preds_dict=predictions, result_networks=["ref"])["ref"]
        for key in ["betas", "betas_limbs", "pose_rotmat", "trans", "flength"]:
            finite_tensor(key, result[key])
    gates["image_inference"] = True
    smal_type = model.smal_model_type
    del model, predictions
    torch.cuda.empty_cache()
    model_config = SMAL_MODEL_CONFIG[smal_type]
    smal = SMAL(smal_model_type=smal_type, template_name="neutral", logscale_part_list=model_config["logscale_part_list"]).to(device)
    keypoint_weights = torch.tensor(data_info.keypoint_weights, device=device)[None, :]
    faces = smal.faces[None]
    remesh_path = source / "data/smal_data_remeshed/uniform_surface_sampling/my_smpl_39dogsnorm_Jr_4_dog_remesh4000_info.pkl"
    with remesh_path.open("rb") as handle:
        remesh = pickle.load(handle, encoding="latin1")
    remesh_faces = torch.as_tensor(remesh["smal_faces"][remesh["faceid_closest"]], device=device, dtype=torch.long)
    remesh_bary = torch.as_tensor(remesh["barys_closest"], device=device, dtype=torch.float32)
    loss_path = source / "src/configs/ttopt_loss_weights/bite_loss_weights_ttopt.json"
    loss_weights = json.loads(loss_path.read_text())
    (output / "loss-weights.json").write_text(json.dumps(loss_weights, indent=2) + "\n")

    def parameter(tensor):
        return tensor.clone().detach().requires_grad_(True)

    local_pose = parameter(rotmat_to_rot6d(result["pose_rotmat"][:, 1:].reshape(-1, 3, 3)).reshape(1, 34, 6))
    orientation = parameter(rotmat_to_rot6d(result["pose_rotmat"][:, :1].reshape(-1, 3, 3)).reshape(1, 1, 6))
    betas = parameter(result["betas"])
    limbs = parameter(result["betas_limbs"])
    trans_xy = parameter(result["trans"][:, :2])
    trans_z = parameter(result["trans"][:, 2:3])
    focal = parameter(result["flength"])
    compact = torch.zeros((1, 2 * smal.n_center + 3 * smal.n_left), device=device, requires_grad=True)
    base_parameters = [focal, trans_z, trans_xy, local_pose, orientation, betas, limbs]
    optimizers = [torch.optim.SGD(base_parameters, lr=0.0025, momentum=0.9), torch.optim.SGD(base_parameters + [compact], lr=0.0001, momentum=0.9)]
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=5) for optimizer in optimizers]
    confidence = result["hg_keyp_scores"].reshape(-1).detach().clone()
    confidence[confidence < 0.2] = 0
    selected = confidence > 0
    target_weights = confidence[selected] * keypoint_weights.reshape(-1)[selected]
    target_keypoints = result["hg_keyp_256"].reshape(-1, 2).detach()
    target_silhouette = result["hg_silh_prep"][0].detach()
    ground_contact = torch.softmax(result["vertexwise_ground_contact"][0], dim=1)[None, :, 1]
    ground_remeshed = torch.einsum("ij,aij->ai", remesh_bary, ground_contact[:, remesh_faces]).round().long()
    isflat = [bool(result["isflat_prep"][0] >= 0.5)]
    touching = [bool(ground_remeshed.sum() > 3)]
    prior_functions = {
        "pose_legs_side": leg_sideway_error,
        "pose_legs_tors": leg_torsion_error,
        "pose_tail_side": tail_sideway_error,
        "pose_tail_tors": tail_torsion_error,
        "pose_spine_side": spine_sideway_error,
        "pose_spine_tors": spine_torsion_error,
    }
    history = []
    started = time.monotonic()
    comparison_vertices = None
    laplacian_comparison = None
    arap = None

    def evaluate_mesh():
        pose = get_optimed_pose_with_glob(orientation, local_pose)
        translation = torch.cat((trans_xy, trans_z), dim=1)
        vertices, points, _ = smal(beta=betas, betas_limbs=limbs, pose=pose, vert_off_compact=compact, trans=translation, keyp_conf="olive", get_skin=True)
        return vertices, points, pose, translation

    for step in range(args.steps):
        stage = int(step >= 150)
        weight_name = "weight_vshift" if stage else "weight"
        if step == 150:
            comparison_vertices = vertices.detach().clone()
            if loss_weights["lapctf"]["weight_vshift"] > 0:
                model_name = Path(SMAL_MODEL_CONFIG["39dogs_norm"]["smal_model_path"]).name.replace(".pkl", "_template.npz")
                data = np.load(source / "data/graphcmr_data" / ("mesh_downsampling_" + model_name), encoding="latin1", allow_pickle=True)
                laplacian_comparison = LaplacianCTF(data["A"][0], device=device)
            if loss_weights["arap"]["weight_vshift"] > 0:
                from combined_model.loss_utils.loss_arap import Arap_Loss
                arap = Arap_Loss(meshes=Meshes(comparison_vertices, faces), device=device)
        optimizer = optimizers[stage]
        optimizer.zero_grad()
        vertices, points, pose, translation = evaluate_mesh()
        silhouette, keypoints = renderer(vertices=vertices, points=points, faces=faces, focal_lengths=focal)
        values = {"silhouette": torch.abs(silhouette[0, 0] - target_silhouette).mean()}
        difference = keypoints[0, :24].reshape(-1, 2) - target_keypoints
        values["keyp"] = (difference[selected].square().sum(dim=1).sqrt() * target_weights).sum() / target_weights.sum().clamp_min(1e-5)
        values.update({name: function(pose) for name, function in prior_functions.items()})
        remeshed_vertices = torch.einsum("ij,aijk->aik", remesh_bary, vertices[:, remesh_faces])
        plane, below_plane = calculate_plane_errors_batch(remeshed_vertices, ground_remeshed, isflat, touching)
        values["gc_plane"] = plane.mean()
        values["gc_belowplane"] = below_plane.mean()
        if any(loss_weights[name][weight_name] > 0 for name in ["edge", "normal", "laplacian", "arap"]):
            mesh = Meshes(vertices, faces)
            if loss_weights["edge"][weight_name] > 0:
                values["edge"] = mesh_edge_loss(mesh)
            if loss_weights["normal"][weight_name] > 0:
                values["normal"] = mesh_normal_consistency(mesh)
            if loss_weights["laplacian"][weight_name] > 0:
                values["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")
            if loss_weights["arap"][weight_name] > 0:
                values["arap"] = arap(mesh)
        if loss_weights["lapctf"][weight_name] > 0:
            values["lapctf"] = laplacian_comparison(vertices, comparison_vertices)[0]
        loss = sum(value * loss_weights[name][weight_name] for name, value in values.items() if loss_weights[name][weight_name] > 0)
        finite_tensor("optimization loss", loss)
        loss.backward()
        for index, parameter_value in enumerate(base_parameters + ([compact] if stage else [])):
            if parameter_value.grad is not None:
                finite_tensor(f"parameter gradient {index}", parameter_value.grad)
        optimizer.step()
        schedulers[stage].step(loss.detach())
        for parameter_value in base_parameters + [compact]:
            finite_tensor("optimized parameters", parameter_value)
        if step == 0:
            gates["finite_optimization_step"] = True
            (output / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
        metrics = {"step": step, "loss": float(loss.detach()), "silhouette": float(values["silhouette"].detach()), "keypoints": float(values["keyp"].detach())}
        history.append(metrics)
        if step % 25 == 0 or step == args.steps - 1:
            print(json.dumps(metrics), flush=True)

    with torch.no_grad():
        vertices, points, pose, translation = evaluate_mesh()
        silhouette, _ = renderer(vertices=vertices, points=points, faces=faces, focal_lengths=focal)
        offsets = full_offsets(smal, compact)
        arrays = {"betas": betas[0], "betas_limbs": limbs[0], "pose_rotmat": pose[0], "trans": translation[0], "flength": focal[0], "offsets": offsets[0], "offsets_compact": compact[0], "verts": vertices[0], "faces": faces[0]}
        arrays = {name: value.detach().cpu().numpy() for name, value in arrays.items()}
        np.savez_compressed(output / "fit.npz", **arrays)
        smal_arrays = {name: getattr(smal, name).detach().cpu().numpy() for name in ["v_template", "shapedirs", "posedirs", "J_regressor", "weights", "betas_scale_mask", "faces"]}
        smal_arrays["parents"] = smal.parents
        np.savez_compressed(output / "smal_model.npz", **smal_arrays)
        identity_pose = torch.eye(3, device=device)[None, None].repeat(1, 35, 1, 1)
        native_vertices, _, _ = smal(beta=betas, betas_limbs=limbs, pose=identity_pose, vert_off_compact=compact, trans=torch.zeros_like(translation), keyp_conf="red", get_skin=True)
        native_joints = smal.J_transformed[0].cpu().numpy()
        rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)
        native = native_vertices[0].cpu().numpy()
        oriented = native @ rotation.T
        span = float(np.ptp(oriented, axis=0).max())
        center = (oriented.max(axis=0) + oriented.min(axis=0)) * 0.5
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation / span
        transform[:3, 3] = -center / span
        canonical = dict(arrays)
        canonical.update(verts=(oriented - center) / span, native_verts=native, pose_rotmat=identity_pose[0].cpu().numpy(), trans=np.zeros(3, dtype=np.float32), coordinate_transform=transform, native_joints=native_joints, joints=(native_joints @ rotation.T - center) / span)
        np.savez_compressed(output / "canonical.npz", **canonical)
        import trimesh
        trimesh.Trimesh(vertices=arrays["verts"], faces=arrays["faces"], process=False).export(output / "fit.obj")
        trimesh.Trimesh(vertices=canonical["verts"], faces=arrays["faces"], process=False).export(output / "canonical.obj")
        crop = input_image[0].detach().clone()
        for channel, mean in zip(crop, data_info.rgb_mean):
            channel.add_(mean)
        crop_rgb = crop.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        mask = silhouette[0, 0].cpu().numpy()
        overlay = crop_rgb.copy()
        overlay[mask > 0.5] = 0.6 * overlay[mask > 0.5] + 0.4 * np.array([0, 0.67, 0.87])
        Image.fromarray(np.uint8(overlay * 255)).save(output / "fit-overlay.png")
        Image.fromarray(np.uint8(mask.clip(0, 1) * 255)).save(output / "fit-silhouette.png")
        Image.fromarray(np.uint8(result["hg_silh_prep"][0].cpu().numpy().clip(0, 1) * 255)).save(output / "predicted-silhouette.png")
        rotations, translations = look_at_view_transform(dist=2.0, elev=12, azim=145, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=rotations, T=translations, fov=40)
        preview_vertices = torch.from_numpy(canonical["verts"]).to(device)[None]
        preview_mesh = Meshes(preview_vertices, faces, textures=TexturesVertex(verts_features=torch.full_like(preview_vertices, 0.75)))
        preview_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=512, blur_radius=0, faces_per_pixel=1)), shader=HardPhongShader(device=device, cameras=cameras, lights=PointLights(device=device, location=[[1, 2, 3]])))
        preview = preview_renderer(preview_mesh)[0, :, :, :3].cpu().numpy().clip(0, 1)
        Image.fromarray(np.uint8(preview * 255)).save(output / "canonical-preview.png")
    gates.update(steps_completed=args.steps, duration_seconds=time.monotonic() - started, final_loss=history[-1]["loss"], final_silhouette=history[-1]["silhouette"])
    (output / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    (output / "optimization.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
