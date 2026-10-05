"""Check cloning, splitting, optimizer state and the density cap."""

import json
import math
import torch
from train import FreeGaussians, density_indices


def main():
    torch.manual_seed(42)
    parameters = {"means": torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), "colors": torch.full((3, 3), 0.5), "quats": torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1), "log_scales": torch.tensor([[0.001] * 3, [0.1] * 3, [0.005] * 3]).log(), "opacity_logits": torch.zeros(3)}
    gaussians = FreeGaussians(parameters)
    optimizer = torch.optim.Adam(gaussians.optimizer_groups())
    sum(value.square().sum() for value in gaussians.values()).backward()
    optimizer.step()
    old = {key: value.detach().clone() for key, value in gaussians.items()}
    moments = {key: optimizer.state[value]["exp_avg"].clone() for key, value in gaussians.items()}
    score = torch.tensor([0.002, 0.003, 0.0001])
    keep, children, split = density_indices(torch.ones(3), score, parameters["log_scales"].exp(), 5)
    assert keep.tolist() == [0, 2]
    assert children.tolist() == [0, 1, 1]
    assert split.tolist() == [False, True, True]
    gaussians.resize(optimizer, keep, children, split)
    assert len(gaussians["means"]) == 5
    assert torch.equal(gaussians["means"][2], old["means"][0])
    assert torch.equal(gaussians["log_scales"][2], old["log_scales"][0])
    assert torch.allclose(gaussians["log_scales"][3:], old["log_scales"][1] - math.log(1.6))
    assert not torch.equal(gaussians["means"][3], old["means"][1])
    for key, value in gaussians.items():
        assert torch.equal(optimizer.state[value]["exp_avg"][:2], moments[key][keep])
        assert torch.count_nonzero(optimizer.state[value]["exp_avg"][2:]) == 0
    keep, children, split = density_indices(torch.ones(3), score, parameters["log_scales"].exp(), 4)
    assert len(keep) + len(children) == 4
    optimizer.zero_grad()
    sum(value.square().mean() for value in gaussians.values()).backward()
    optimizer.step()
    assert all(torch.isfinite(value).all() for value in gaussians.values())
    print(json.dumps({"status": "passed", "checks": ["exact_clone", "two_child_split", "parent_removal", "scale_reduction", "adam_state_retention", "new_state_zero", "cap", "next_optimizer_step"]}))


if __name__ == "__main__":
    main()
