"""Scientific checks: amplitude invariance, exact modes, zero-field handling."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifact_tools import evaluate


class SineMode(torch.nn.Module):
    def __init__(self, dim, amplitude):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(amplitude)))
        self.dim = dim

    def forward(self, points):
        u = torch.sin(points[:, 0])
        if self.dim == 2:
            u = u * torch.sin(points[:, 1])
        return (self.scale * u)[:, None]


class ArtifactTests(unittest.TestCase):
    def config(self, dim):
        path = Path(__file__).resolve().parents[1] / f"laplace_{dim}d/config/config.yaml"
        cfg = OmegaConf.load(path)
        cfg.evaluation.spatial_points_per_axis = 9
        cfg.evaluation.lambda_points = 7
        cfg.evaluation.chunk_size = 13
        return cfg

    def evaluation(self, dim, amplitude):
        with tempfile.TemporaryDirectory() as directory:
            return evaluate(SineMode(dim, amplitude), self.config(dim), Path(directory) / "data.npz")

    def test_exact_modes_and_relative_residual(self):
        for dim in (1, 2):
            with self.subTest(dim=dim):
                data = self.evaluation(dim, 2.0)
                # sin(x), sin(x)sin(y) have eigenvalues 1 and 2 respectively.
                expected = (data["lambdas"] - dim) ** 2
                np.testing.assert_allclose(data["relative_pde_mse"], expected, atol=2e-5, rtol=2e-5)
                index = np.argmin(abs(data["lambdas"] - dim))
                self.assertLess(data["pde_mse"][index], 1e-10)
                self.assertLess(data["bc_mse"].max(), 1e-10)

    def test_small_amplitude_does_not_improve_relative_residual(self):
        for dim in (1, 2):
            with self.subTest(dim=dim):
                big, small = self.evaluation(dim, 2), self.evaluation(dim, .002)
                np.testing.assert_allclose(big["relative_pde_mse"], small["relative_pde_mse"],
                                           atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(big["unit_peak"], small["unit_peak"], atol=1e-6)
                np.testing.assert_allclose(np.max(abs(small["unit_peak"]), axis=1), 1)
                np.testing.assert_allclose(small["energy"], big["energy"] * 1e-6, rtol=1e-5)

    def test_zero_field_is_not_a_valid_normalized_mode(self):
        data = self.evaluation(1, 0)
        self.assertFalse(data["valid"].any())
        self.assertTrue(np.isnan(data["unit_peak"]).all())
        self.assertTrue(np.isnan(data["relative_pde_mse"]).all())
        self.assertTrue(np.isnan(data["relative_bc_mse"]).all())


if __name__ == "__main__":
    unittest.main()
