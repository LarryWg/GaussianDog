"""Optimize the authorized upstream D-SMAL model without truncating skin weights."""

from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation


def axis_angle_matrix(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(angle / torch.pi)
    b = 0.5 * torch.sinc(angle / (2 * torch.pi)).square()
    return torch.eye(3, dtype=vector.dtype, device=vector.device) + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def compact_offsets(smal, compact):
    compact = compact.reshape(-1)
    nc = smal.n_center
    nl = smal.n_left
    result = np.zeros((3889, 3), dtype=np.float32)
    ids = smal.sym_ids_dict
    result[ids["center"], 0] = compact[:nc]
    result[ids["center"], 2] = compact[nc:2 * nc]
    result[ids["left"]] = compact[2 * nc:].reshape(3, nl).T
    result[ids["right"]] = result[ids["left"]] * [1, -1, 1]
    return result


class PetModel(nn.Module):
    def __init__(self, bite_source, fit_path, device="cuda"):
        super().__init__()
        sys.path.insert(0, str(Path(bite_source).resolve() / "src"))
        from smal_pytorch.smal_model.smal_torch_new import SMAL

        self.smal = SMAL(smal_model_type="39dogs_norm_newv3", template_name="neutral", logscale_part_list=["legs_l", "legs_f", "tail_l", "tail_f", "ears_y", "ears_l", "head_l"]).to(device)
        fit = np.load(fit_path)
        keys = set(fit.files)
        betas = fit["betas"].reshape(-1)[:30]
        limbs = fit["betas_limbs"] if "betas_limbs" in keys else fit["betas_limb"]
        pose = fit["pose_rotmat"] if "pose_rotmat" in keys else fit["pose"]
        pose = pose.reshape(35, 3, 3)
        offsets = fit["offsets"].reshape(3889, 3) if "offsets" in keys else compact_offsets(self.smal, fit["vert_off_compact"]) if "vert_off_compact" in keys else np.zeros((3889, 3))
        self.betas = nn.Parameter(torch.as_tensor(betas, dtype=torch.float32, device=device)[None])
        self.limbs = nn.Parameter(torch.as_tensor(limbs, dtype=torch.float32, device=device).reshape(1, 7))
        self.pose = nn.Parameter(torch.as_tensor(Rotation.from_matrix(pose).as_rotvec(), dtype=torch.float32, device=device)[None])
        self.offsets = nn.Parameter(torch.as_tensor(offsets, dtype=torch.float32, device=device)[None])
        self.translation = nn.Parameter(torch.zeros(3, device=device))
        self.log_scale = nn.Parameter(torch.zeros((), device=device))
        self.register_buffer("alignment", torch.eye(3, device=device))
        self.register_buffer("initial_pose", self.pose.detach().clone())
        self.register_buffer("initial_offsets", self.offsets.detach().clone())
        self.register_buffer("faces", self.smal.faces.long())
        self.register_buffer("edge_pairs", torch.unique(torch.sort(torch.cat((self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]])), dim=1).values, dim=0))
        self.register_buffer("initial_edges", torch.zeros(len(self.edge_pairs), device=device))
        self.capture_regularizers()

    def forward(self, pose=None, translation=None):
        rotations = axis_angle_matrix(self.pose if pose is None else pose)
        vertices, _, _ = self.smal(beta=self.betas, betas_limbs=self.limbs, pose=rotations, del_v=self.offsets, trans=torch.zeros((1, 3), device=self.betas.device), get_skin=True)
        vertices = (vertices[0] @ self.alignment.T) * self.log_scale.exp() + self.translation
        return vertices if translation is None else vertices + translation

    def set_alignment(self, rotation, scale, translation):
        with torch.no_grad():
            self.alignment.copy_(torch.as_tensor(rotation, dtype=torch.float32, device=self.betas.device))
            self.log_scale.copy_(torch.as_tensor(np.log(scale), device=self.betas.device))
            self.translation.copy_(torch.as_tensor(translation, dtype=torch.float32, device=self.betas.device))
        self.capture_regularizers()

    def capture_regularizers(self):
        with torch.no_grad():
            vertices = self()
            self.initial_edges.copy_(torch.linalg.vector_norm(vertices[self.edge_pairs[:, 0]] - vertices[self.edge_pairs[:, 1]], dim=-1))

    def regularization(self, vertices):
        edge = vertices[self.edge_pairs[:, 0]] - vertices[self.edge_pairs[:, 1]]
        edge_loss = (torch.linalg.vector_norm(edge, dim=-1) - self.initial_edges).square().mean()
        sums = torch.zeros_like(vertices)
        degrees = torch.zeros(len(vertices), device=vertices.device)
        for side in (0, 1):
            source = self.edge_pairs[:, side]
            target = self.edge_pairs[:, 1 - side]
            sums.index_add_(0, source, vertices[target])
            degrees.index_add_(0, source, torch.ones(len(source), device=vertices.device))
        lap = torch.linalg.vector_norm(vertices - sums / degrees.clamp_min(1)[:, None], dim=-1).mean()
        joints = list(range(7, 15)) + list(range(17, 25)) + [32]
        pose = (self.pose[:, joints] - self.initial_pose[:, joints]).square().sum()
        offsets = self.offsets.square().sum()
        return {"edge": edge_loss, "laplacian": lap, "pose": pose, "offsets": offsets}

    def optimizer_groups(self, factor=1.0):
        return [{"params": [self.betas], "lr": 0.002 * factor, "name": "shape"}, {"params": [self.limbs], "lr": 0.001 * factor, "name": "limbs"}, {"params": [self.pose, self.translation, self.log_scale, self.offsets], "lr": 0.0005 * factor, "name": "pose"}]

    def export_fit(self, path):
        np.savez(path, betas=self.betas.detach().cpu().numpy(), betas_limbs=self.limbs.detach().cpu().numpy(), pose_rotmat=axis_angle_matrix(self.pose).detach().cpu().numpy(), offsets=self.offsets.detach().cpu().numpy(), alignment=self.alignment.cpu().numpy(), global_scale=self.log_scale.detach().exp().cpu().numpy(), translation=self.translation.detach().cpu().numpy(), vertices=self().detach().cpu().numpy(), faces=self.faces.cpu().numpy())
