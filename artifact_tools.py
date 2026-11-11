"""Numeric artifacts and postprocessing only; each experiment owns its PDE/train/config."""
import csv
import hashlib
import importlib.metadata
import inspect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def seed_everything(seed, requested_device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device == "auto":
        device = "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    # Legacy MultiPINN uses torch.Tensor(...) in Condition and adaptive weights.
    # set_default_device alone does not move those legacy constructors to CUDA.
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def reference_modes(cfg, spatial):
    """Known analytical modes are used only for evaluation, never training."""
    dim = cfg.problem.spatial_dim
    if cfg.problem.name == "heat_continuous":
        return np.empty(0), np.empty((0, len(spatial))), np.empty((0, dim), dtype=int)
    eigs, modes, indices = [], [], []
    max_index = int(np.sqrt(max(0, cfg.problem.lambda_max))) + 1
    for n in range(1, max_index + 1):
        for m in (range(1, max_index + 1) if dim == 2 else [0]):
            eigenvalue = (np.pi / cfg.problem.x_max) ** 2 * (n * n + m * m)
            if cfg.problem.lambda_min <= eigenvalue <= cfg.problem.lambda_max:
                mode = np.sin(n * np.pi * spatial[:, 0] / cfg.problem.x_max)
                if dim == 2:
                    mode = mode * np.sin(m * np.pi * spatial[:, 1] / cfg.problem.x_max)
                eigs.append(eigenvalue)
                modes.append(mode)
                indices.append([n, m] if dim == 2 else [n])
    return np.asarray(eigs), np.asarray(modes).reshape(-1, len(spatial)), np.asarray(indices, dtype=int).reshape(-1, dim)


def evaluation_grid(cfg):
    count = cfg.evaluation.spatial_points_per_axis
    if count < 3 or cfg.evaluation.lambda_points < 2:
        raise ValueError("Evaluation requires >=3 spatial points/axis and >=2 lambda points.")
    axis = np.linspace(0, cfg.problem.x_max, count)
    mesh = np.meshgrid(*([axis] * cfg.problem.spatial_dim), indexing="ij")
    spatial = np.stack([a.ravel() for a in mesh], axis=1)
    eigs, modes, indices = reference_modes(cfg, spatial)
    lambdas = np.unique(np.concatenate([
        np.linspace(cfg.problem.lambda_min, cfg.problem.lambda_max, cfg.evaluation.lambda_points),
        eigs,
    ]))
    return spatial, lambdas, eigs, modes, indices


def predict_residual(model, points, spatial_dim, chunk_size):
    """Pointwise second derivatives, bounded evaluation memory, no parameter gradients."""
    parameter = next(model.parameters())
    predictions, residuals = [], []
    for start in range(0, len(points), chunk_size):
        with torch.enable_grad():
            p = torch.as_tensor(points[start:start + chunk_size], device=parameter.device,
                                dtype=parameter.dtype).requires_grad_(True)
            u = model(p)[:, 0]
            first = torch.autograd.grad(u.sum(), p, create_graph=True)[0]
            laplacian = torch.zeros_like(u)
            for axis in range(spatial_dim):
                second = torch.autograd.grad(first[:, axis].sum(), p, retain_graph=True)[0]
                laplacian = laplacian + second[:, axis]
            residual = laplacian + p[:, -1] * u
        predictions.append(u.detach().cpu().numpy())
        residuals.append(residual.detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(residuals)


def evaluate(model, cfg, destination):
    """Independent fixed grid, raw/unit-peak fields and amplitude-aware diagnostics."""
    spatial, lambdas, eigenvalues, references, mode_indices = evaluation_grid(cfg)
    dim = cfg.problem.spatial_dim
    boundary_masks = [np.isclose(spatial[:, 0], 0)]
    if cfg.problem.name != "heat_continuous":
        boundary_masks.append(np.isclose(spatial[:, 0], cfg.problem.x_max))
    if dim == 2:
        boundary_masks.extend([np.isclose(spatial[:, 1], 0),
                               np.isclose(spatial[:, 1], cfg.problem.x_max)])
    interior = np.all((spatial > 0) & (spatial < cfg.problem.x_max), axis=1)
    raw, pde, energies, peaks, bc = [], [], [], [], []
    was_training = model.training
    model.eval()
    try:
        for lam in lambdas:
            points = np.column_stack([spatial, np.full(len(spatial), lam)])
            values, residual = predict_residual(model, points, dim, cfg.evaluation.chunk_size)
            raw.append(values)
            pde.append(np.mean(residual[interior].astype(np.float64) ** 2))
            energies.append(np.mean(values[interior].astype(np.float64) ** 2))
            peaks.append(np.max(np.abs(values)))
            bc.append([np.mean(values[mask].astype(np.float64) ** 2) for mask in boundary_masks])
    finally:
        model.train(was_training)
    raw, pde, energy, peak, bc = map(np.asarray, (raw, pde, energies, peaks, bc))
    rms = np.sqrt(energy)
    valid = (rms > cfg.evaluation.min_rms) & (peak > cfg.evaluation.min_peak)
    normalized = np.full_like(raw, np.nan)
    normalized[valid] = raw[valid] / peak[valid, None]
    relative_pde = np.full_like(pde, np.nan)
    relative_bc = np.full_like(bc, np.nan)
    relative_pde[valid] = pde[valid] / energy[valid]
    relative_bc[valid] = bc[valid] / energy[valid, None]
    data = dict(
        spatial=spatial, lambdas=lambdas, raw=raw, unit_peak=normalized,
        energy=energy, rms=rms, peak=peak, valid=valid, pde_mse=pde,
        relative_pde_mse=relative_pde, bc_mse=bc, relative_bc_mse=relative_bc,
        normalization_loss=(peak - 1) ** 2,
        nontriviality_loss=(cfg.problem.nontriviality_a * energy + cfg.problem.nontriviality_b) ** -2,
        reference_eigenvalues=eigenvalues, reference_modes=references,
        reference_mode_indices=mode_indices,
    )
    if cfg.problem.name == "heat_continuous":
        data["reference_field"] = np.where(
            lambdas[:, None] > 0,
            np.sin(np.sqrt(np.abs(lambdas[:, None])) * spatial[None, :, 0]), 0,
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **data)
    return data


def runtime_metadata(regularization=None):
    from multipinn import Trainer
    from multipinn.neural_network import PirateNet
    from multipinn.regularization import AdaptiveConditionsLosses
    sources = {}
    classes = [Trainer, PirateNet, AdaptiveConditionsLosses]
    if regularization is not None:
        classes.append(type(regularization))
    for cls in classes:
        path = Path(inspect.getfile(cls))
        sources[cls.__name__] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    versions = {}
    for package in ("torch", "numpy", "hydra-core", "multipinn", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source checkout / not installed as distribution"
    return {"python": sys.version, "versions": versions, "sources": sources,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}


class TrainingArtifacts:
    """Duck-typed non-gradient MultiPINN callback; never draws figures."""
    def __init__(self, cfg, names, regularization):
        self.cfg, self.names, self.regularization = cfg, names, regularization
        self.root = Path(cfg.paths.save_dir)
        (self.root / "checkpoints").mkdir(exist_ok=True)
        (self.root / "predictions").mkdir(exist_ok=True)
        self.stream = (self.root / "metrics.csv").open("x", newline="")
        fields = ["epoch", "seconds", "lr", "total_loss"]
        fields += ["loss_" + n for n in names] + ["weight_" + n for n in names]
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()
        self.started = time.monotonic()
        (self.root / "environment.json").write_text(json.dumps(runtime_metadata(regularization), indent=2))

    def __call__(self, trainer):
        epoch = trainer.current_epoch + 1
        adaptive = self.regularization.adaptive_cond
        weights = adaptive.weight.detach().cpu().tolist()
        losses = trainer.epoch_loss_detailed.detach().cpu().tolist()
        row = dict(epoch=epoch, seconds=time.monotonic() - self.started,
                   lr=trainer.current_lr, total_loss=float(trainer.total_loss))
        row.update({"loss_" + n: v for n, v in zip(self.names, losses)})
        row.update({"weight_" + n: v for n, v in zip(self.names, weights)})
        self.writer.writerow(row)
        if epoch == 1 or epoch % 100 == 0:
            self.stream.flush()
            print(f"epoch={epoch} loss={row['total_loss']:.6g} lr={row['lr']:.3g}", flush=True)
        if epoch % self.cfg.artifacts.save_period == 0 or epoch == self.cfg.trainer.num_epochs:
            self.stream.flush()
            config = OmegaConf.to_container(self.cfg, resolve=True)
            numpy_state = np.random.get_state()
            checkpoint = dict(
                epoch=epoch, config=config,
                model_state_dict=trainer.pinn.model.state_dict(),
                optimizer_state_dict=trainer.optimizer.state_dict(),
                scheduler_state_dict=trainer.scheduler.state_dict(),
                adaptive_logits=adaptive.train_weight.detach(),
                adaptive_optimizer_state_dict=adaptive.optimizer.state_dict(),
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python_rng_state=random.getstate(),
                numpy_rng_state=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                collocation_points=[c.points.detach() for c in trainer.pinn.conditions],
                condition_names=self.names,
            )
            torch.save(checkpoint, self.root / "checkpoints" / f"epoch_{epoch:07d}.pth")
            evaluate(trainer.pinn.model, self.cfg,
                     self.root / "predictions" / f"epoch_{epoch:07d}.npz")

    def close(self):
        self.stream.close()
