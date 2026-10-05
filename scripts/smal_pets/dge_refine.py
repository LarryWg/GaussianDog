"""Run one bounded official DGE fur edit on a calibrated SMAL-pets cloud."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from gaussians import read_ply


MODEL = "timbrooks/instruct-pix2pix"
MODEL_REVISION = "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"
UPSTREAM = "https://github.com/silent-chen/DGE"
UPSTREAM_REVISION = "7eed0bc57824707c53b935bd19ea994f19f41eae"
STEPS = 1000
VIEW_COUNT = 20
SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cloud(path, cap):
    fields = PlyData.read(path)["vertex"].data.dtype.names
    if any(name.startswith("f_rest_") for name in fields) or not all(f"f_dc_{axis}" in fields for axis in range(3)):
        raise ValueError("DGE input and output must use SH0 Gaussian colors")
    parameters = read_ply(path)
    count = len(parameters["means"])
    if not 0 < count <= cap:
        raise ValueError(f"Gaussian count {count} exceeds the bounded range 1..{cap}")
    if any(not np.isfinite(value).all() for value in parameters.values()):
        raise ValueError("Gaussian parameters contain non-finite values")
    if np.any(np.linalg.norm(parameters["quats"], axis=1) <= 1e-8):
        raise ValueError("Gaussian orientations contain a zero quaternion")
    return parameters


def select_views(configuration):
    views = [view for view in configuration["views"] if view.get("split") == "train"]
    if configuration["resolution"] != 512 or len(views) < VIEW_COUNT:
        raise ValueError("DGE needs at least 20 calibrated 512px training views")
    centers = []
    for view in views:
        camera = np.asarray(view["view"], dtype=np.float64)
        intrinsic = np.asarray(view["K"], dtype=np.float64)
        if camera.shape != (4, 4) or intrinsic.shape != (3, 3):
            raise ValueError("Invalid calibrated camera matrix dimensions")
        if not np.isfinite(camera).all() or not np.isfinite(intrinsic).all():
            raise ValueError("Camera calibration contains non-finite values")
        rotation = camera[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Camera rotation is not a proper orthonormal frame")
        if not np.allclose(camera[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("Camera must use a homogeneous OpenCV world-to-camera transform")
        expected = np.diag([intrinsic[0, 0], intrinsic[1, 1], 1])
        expected[:2, 2] = 256
        if min(intrinsic[0, 0], intrinsic[1, 1]) <= 0 or not np.allclose(intrinsic, expected, atol=1e-5):
            raise ValueError("DGE requires centered undistorted PINHOLE intrinsics")
        centers.append(-rotation.T @ camera[:3, 3])
    centers = np.asarray(centers)
    chosen = [int(np.random.default_rng(SEED).integers(len(views)))]
    distances = np.full(len(views), np.inf)
    while len(chosen) < VIEW_COUNT:
        distances = np.minimum(distances, np.linalg.norm(centers - centers[chosen[-1]], axis=1))
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return [views[index] for index in chosen]


def make_colmap(training, output, parameters):
    configuration = json.loads((training / "cameras.json").read_text())
    views = select_views(configuration)
    scene = output / "colmap"
    images = scene / "images"
    sparse = scene / "sparse/0"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    cameras = ["# CAMERA_ID MODEL WIDTH HEIGHT FX FY CX CY"]
    extrinsics = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME"]
    for index, view in enumerate(views, start=1):
        name = f"{index:04d}.png"
        image = Image.open(training / view["file"]).convert("RGB")
        if image.size != (512, 512):
            raise ValueError(f"Calibrated image has the wrong resolution: {view['file']}")
        image.save(images / name)
        intrinsic = np.asarray(view["K"])
        camera = np.asarray(view["view"])
        quaternion = Rotation.from_matrix(camera[:3, :3]).as_quat()[[3, 0, 1, 2]]
        cameras.append(f"{index} PINHOLE 512 512 {intrinsic[0, 0]:.17g} {intrinsic[1, 1]:.17g} 256 256")
        pose = " ".join(f"{value:.17g}" for value in np.r_[quaternion, camera[:3, 3]])
        extrinsics.extend([f"{index} {pose} {index} {name}", ""])
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (sparse / "images.txt").write_text("\n".join(extrinsics) + "\n")
    count = len(parameters["means"])
    cloud = np.zeros(count, dtype=[(key, "<f4") for key in ["x", "y", "z", "nx", "ny", "nz"]] + [(key, "u1") for key in ["red", "green", "blue"]])
    for axis, name in enumerate(["x", "y", "z"]):
        cloud[name] = parameters["means"][:, axis]
    for axis, name in enumerate(["red", "green", "blue"]):
        cloud[name] = np.round(np.clip(parameters["colors"][:, axis], 0, 1) * 255).astype(np.uint8)
    PlyData([PlyElement.describe(cloud, "vertex")], text=False).write(sparse / "points3D.ply")
    (scene / "calibration.json").write_text(json.dumps({"coordinateConvention": "OpenCV world-to-camera", "background": [1, 1, 1], "imagePreparation": "preserve-calibrated-white-RGB", "selectedViews": views}, indent=2) + "\n")
    return scene, [view["id"] for view in views]


def replace(source, relative, old, new):
    path = source / relative
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"DGE compatibility target changed: {relative}")
    path.write_text(text.replace(old, new))


def adapt_upstream(source):
    replace(source, "threestudio/utils/misc.py", "import tinycudann as tcnn", "try:\n    import tinycudann as tcnn\nexcept ImportError:\n    tcnn = None")
    replace(source, "threestudio/utils/misc.py", "    tcnn.free_temporary_memory()", "    if tcnn is not None:\n        tcnn.free_temporary_memory()")
    replace(source, "threestudio/models/exporters/__init__.py", "from . import base, mesh_exporter\n", "from . import base\n")
    replace(source, "threestudio/utils/ops.py", "from igl import fast_winding_number_for_meshes, point_mesh_squared_distance, read_obj", "try:\n    from igl import fast_winding_number_for_meshes, point_mesh_squared_distance, read_obj\nexcept ImportError:\n    pass")
    replace(source, "threestudio/systems/DGE.py", "from threestudio.utils.sam import LangSAMTextSegmentor", "# Global editing does not require a segmentation model.")
    replace(source, "threestudio/systems/DGE.py", "        self.text_segmentor = LangSAMTextSegmentor().to(get_device())", "        self.text_segmentor = None\n        if self.cfg.seg_prompt:\n            from threestudio.utils.sam import LangSAMTextSegmentor\n            self.text_segmentor = LangSAMTextSegmentor().to(get_device())")
    replace(source, "threestudio/systems/DGE.py", "bg_color = [1, 1, 1] if False else [0, 0, 0]\n", "bg_color = [1, 1, 1]\n")
    replace(source, "threestudio/systems/DGE.py", "np.arccos(np.dot(most_left_vecotr, cam.R[:, 2]))", "np.arccos(np.clip(np.dot(most_left_vecotr, cam.R[:, 2]), -1, 1))")


def make_config(source, scene, snapshot, output, prompt, model_path):
    from omegaconf import OmegaConf

    configuration = OmegaConf.load(source / "configs/dge.yaml")
    configuration.name = "smal-fur"
    configuration.tag = "seed42"
    configuration.seed = SEED
    configuration.use_timestamp = False
    configuration.exp_root_dir = str(output / "runs")
    configuration.data.source = str(scene)
    configuration.data.max_view_num = VIEW_COUNT
    configuration.data.height = configuration.data.width = 512
    configuration.data.eval_height = configuration.data.eval_width = 512
    configuration.data.use_original_resolution = True
    configuration.system.gs_source = str(snapshot)
    configuration.system.camera_update_per_step = STEPS + 1
    configuration.system.cache_dir = str(output / "cache")
    configuration.system.prompt_processor.pretrained_model_name_or_path = str(model_path)
    configuration.system.prompt_processor.prompt = prompt
    configuration.system.guidance.ip2p_name_or_path = str(model_path)
    configuration.system.guidance.ddim_scheduler_name_or_path = str(model_path)
    configuration.system.guidance.guidance_scale = 7.5
    configuration.system.guidance.condition_scale = 1.5
    configuration.system.guidance.diffusion_steps = 20
    configuration.system.guidance.camera_batch_size = 5
    configuration.trainer.max_steps = STEPS
    configuration.trainer.precision = "32-true"
    path = output / "dge.yaml"
    OmegaConf.save(configuration, path)
    return path


def worker(source, configuration, cap):
    import torch

    if not torch.cuda.is_available() or not torch.__version__.startswith("2.4.") or torch.version.cuda != "11.8":
        raise RuntimeError("DGE requires the isolated PyTorch 2.4 CUDA 11.8 environment")
    os.chdir(source)
    sys.path.insert(0, str(source))
    from gaussiansplatting.scene.gaussian_model import GaussianModel

    densify = GaussianModel.densify_and_prune

    def bounded_densify(self, *args, **kwargs):
        densify(self, *args, **kwargs)
        count = len(self.get_xyz)
        if count > cap:
            order = torch.argsort(self.get_opacity[:, 0], descending=True)
            prune = torch.ones(count, dtype=torch.bool, device=self.get_xyz.device)
            prune[order[:cap]] = False
            self.remove_grad_mask()
            self.prune_points(prune)
            self.apply_grad_mask(self.mask)
            self.update_anchor()

    GaussianModel.densify_and_prune = bounded_densify
    sys.argv = [str(source / "launch.py"), "--config", str(configuration), "--train", "--gpu", "0"]
    runpy.run_path(str(source / "launch.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--dge-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diffusion-model", type=Path, help="Pinned local InstructPix2Pix directory, defaults to the training parent's dge-model")
    parser.add_argument("--gaussian-cap", type=int, default=150000)
    parser.add_argument("--prompt", default="Make this tricolor dog's coat naturally fluffy with detailed soft long fur. Give its tail black fluffy fur. Preserve its face, body shape, pose, white chest, white facial blaze and white paws, tan eyebrows and cheeks, black coat and black ears, and the plain white background.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = args.dge_source.resolve()
    if not 0 < args.gaussian_cap <= 150000:
        parser.error("Gaussian cap must be between 1 and 150000")
    if args.worker_config:
        worker(source, args.worker_config.resolve(), args.gaussian_cap)
        return
    if not args.training:
        parser.error("--training is required")
    training = args.training.resolve()
    model_path = (args.diffusion_model or training.parent / "dge-model").resolve()
    provenance = json.loads((model_path / "model-provenance.json").read_text())
    if provenance.get("model") != MODEL or provenance.get("revision") != MODEL_REVISION:
        raise RuntimeError("DGE needs the pinned InstructPix2Pix model cache")
    if not (model_path / "model_index.json").is_file():
        raise RuntimeError("DGE diffusion model cache is incomplete")
    training_metadata = json.loads((training / "training.json").read_text())
    if not args.prepare_only and not (training_metadata.get("complete") and training_metadata.get("stage") == "free" and training_metadata.get("bound_steps") == 15000 and training_metadata.get("free_steps") == 25000 and training_metadata.get("step", 0) >= 25000):
        raise RuntimeError("DGE requires completed 15000 bound and 25000 free fitting steps")
    output = (args.output or training / "dge").resolve()
    output.mkdir(parents=True, exist_ok=True)
    refined = output / "refined.ply"
    if refined.exists():
        raise RuntimeError("The completed DGE output already exists")
    original = training / "latest.ply"
    parameters = validate_cloud(original, args.gaussian_cap)
    snapshot = output / "pre-dge.ply"
    if snapshot.exists() and sha256(snapshot) != sha256(original):
        raise RuntimeError("Existing pre-DGE snapshot belongs to a different trained cloud")
    if not snapshot.exists():
        shutil.copy2(original, snapshot)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_REVISION:
        raise RuntimeError("DGE checkout does not match the pinned upstream revision")
    adapt_upstream(source)
    scene, views = make_colmap(training, output, parameters)
    configuration = make_config(source, scene, snapshot, output, args.prompt, model_path)
    report = {"upstream": UPSTREAM, "commit": commit, "model": MODEL, "modelRevision": MODEL_REVISION, "modelPath": str(model_path), "prompt": args.prompt, "seed": SEED, "steps": STEPS, "diffusionSteps": 20, "textGuidance": 7.5, "imageGuidance": 1.5, "viewIds": views, "resolution": 512, "background": [1, 1, 1], "imagePreparation": "preserve-calibrated-white-RGB", "gaussianCap": args.gaussian_cap, "shDegree": 0, "inputCount": len(parameters["means"]), "inputSha256": sha256(snapshot), "status": "prepared"}
    report["sourceTraining"] = {key: training_metadata[key] for key in ("stage", "step", "bound_steps", "free_steps", "complete")}
    report["modelProvenanceSha256"] = sha256(model_path / "model-provenance.json")
    report["systemSourceSha256"] = sha256(source / "threestudio/systems/DGE.py")
    report_path = output / "result.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(report, indent=2))
        return
    command = [sys.executable, str(Path(__file__).resolve()), "--dge-source", str(source), "--worker-config", str(configuration), "--gaussian-cap", str(args.gaussian_cap)]
    started = time.monotonic()
    try:
        subprocess.run(command, check=True)
        result = output / "runs/smal-fur/seed42/save/last.ply"
        edited = output / "runs/smal-fur/seed42/save/edited_images.png"
        checkpoint = output / "runs/smal-fur/seed42/ckpts/last.ckpt"
        if not result.exists() or not edited.exists() or not checkpoint.exists():
            raise RuntimeError("DGE did not export the refined cloud and edited view montage")
        import torch

        completed_steps = int(torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)["global_step"])
        if completed_steps != STEPS:
            raise RuntimeError(f"DGE completed {completed_steps} steps instead of {STEPS}")
        result_parameters = validate_cloud(result, args.gaussian_cap)
        changed = len(result_parameters["means"]) != len(parameters["means"]) or any(not np.array_equal(result_parameters[key], value) for key, value in parameters.items())
        if not changed:
            raise RuntimeError("DGE returned an unchanged Gaussian cloud")
        if sha256(original) != report["inputSha256"]:
            raise RuntimeError("The trained source cloud changed during DGE")
        shutil.copy2(result, refined)
        report.update(status="completed", completedSteps=completed_steps, parametersChanged=changed, sourceUnchanged=True, outputCount=len(result_parameters["means"]), outputSha256=sha256(refined), output=str(refined), editedViews=str(edited))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsedSeconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
