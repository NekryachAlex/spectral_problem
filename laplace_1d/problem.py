"""1D Laplace localization reconstructed from ARTICLE_MATERIALS.md."""
import torch
from multipinn.condition import Condition
from multipinn.geometry import Hypercube


def build_problem(cfg):
    dim = 1
    low = [0.0] * dim + [cfg.lambda_min]
    high = [cfg.x_max] * dim + [cfg.lambda_max]

    def inner(model, points):
        u = model(points)[:, 0]
        first = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
        laplacian = torch.zeros_like(u)
        for axis in range(dim):
            second = torch.autograd.grad(
                first[:, axis].sum(), points, create_graph=True
            )[0][:, axis]
            laplacian = laplacian + second
        return [laplacian + points[:, -1] * u]

    def nontriviality(model, points):
        energy = model(points).square().mean()
        return [(1.0 / (cfg.nontriviality_a * energy + cfg.nontriviality_b)).reshape(1)]

    def normalization(model, points):
        return [(model(points).abs().max() - 1.0).abs().reshape(1)]

    def zero_boundary(model, points):
        return [model(points)[:, 0]]

    domain = Hypercube(low, high)
    conditions = [
        Condition(inner, domain),
        Condition(nontriviality, domain),
        Condition(normalization, domain),
    ]
    names = ["pde", "nontriviality", "normalization"]
    for axis, value, label in [(0, 0.0, "left"), (0, cfg.x_max, "right")]:
        bc_low, bc_high = low.copy(), high.copy()
        bc_low[axis] = bc_high[axis] = value
        conditions.append(Condition(zero_boundary, Hypercube(bc_low, bc_high)))
        names.append("bc_" + label)
    return conditions, names, dim + 1
