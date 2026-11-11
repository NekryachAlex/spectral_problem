"""Recompute numeric predictions/diagnostics from an explicitly selected checkpoint."""
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from multipinn.neural_network import PirateNet

from artifact_tools import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spatial-points", type=int)
    parser.add_argument("--lambda-points", type=int)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    cfg = OmegaConf.create(checkpoint["config"])
    if args.spatial_points:
        cfg.evaluation.spatial_points_per_axis = args.spatial_points
    if args.lambda_points:
        cfg.evaluation.lambda_points = args.lambda_points
    model = PirateNet(input_dim=cfg.problem.spatial_dim + 1, output_dim=1,
                      **cfg.model.params).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    output = args.output or args.checkpoint.parents[1] / "predictions" / (args.checkpoint.stem + ".npz")
    evaluate(model, cfg, output)
    print(output)


if __name__ == "__main__":
    main()
