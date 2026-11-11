"""Standalone Hydra entry point for laplace_2d; no plotting during training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from multipinn import PINN, Generator, Trainer, CallbacksOrganizer
from multipinn.neural_network import PirateNet
from adaptive_conditions import EqualInitialAdaptiveConditionsLosses

from problem import build_problem
from artifact_tools import TrainingArtifacts, seed_everything


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    if cfg.trainer.num_epochs < 1 or cfg.trainer.num_batches != 1:
        raise ValueError("Use num_epochs >= 1 and one full batch: global penalties depend on it.")
    if cfg.generator.domain_points < 1 or cfg.generator.boundary_points_per_condition < 1:
        raise ValueError("Collocation point counts must be positive.")
    if cfg.trainer.grid_update < 1 or cfg.artifacts.save_period < 1:
        raise ValueError("Update/save periods must be positive.")
    device = seed_everything(cfg.trainer.random_seed, cfg.trainer.device)
    save_dir = Path(cfg.paths.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "metrics.csv").exists():
        raise FileExistsError(f"Run already exists: {save_dir}")
    OmegaConf.save(cfg, save_dir / "used_config.yaml", resolve=True)
    conditions, names, input_dim = build_problem(cfg.problem)
    model = PirateNet(input_dim=input_dim, output_dim=1, **cfg.model.params).to(device)
    domain = Generator(cfg.generator.domain_points, sampler=cfg.generator.sampler)
    boundary = Generator(
        cfg.generator.boundary_points_per_condition, sampler=cfg.generator.sampler
    )
    domain.use_for(conditions[:3])
    boundary.use_for(conditions[3:])
    pinn = PINN(model=model, conditions=conditions)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.scheduler.gamma
    )
    if cfg.regularization.type != "AdaptiveConditions":
        raise ValueError("This baseline uses AdaptiveConditions.")
    if cfg.regularization.initialization != "equal":
        raise ValueError("This baseline uses equal initial condition weights.")
    regularization = EqualInitialAdaptiveConditionsLosses(lr=cfg.regularization.lr)
    artifacts = TrainingArtifacts(cfg, names, regularization)
    trainer = Trainer(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        # MultiPINN includes index zero: N requested epochs require indices 0..N-1.
        num_epochs=cfg.trainer.num_epochs - 1,
        num_batches=1,
        update_grid_every=cfg.trainer.grid_update,
        calc_loss=regularization,
        callbacks_organizer=CallbacksOrganizer([artifacts]),
    )
    try:
        trainer.train()
    finally:
        artifacts.close()


if __name__ == "__main__":
    train()
