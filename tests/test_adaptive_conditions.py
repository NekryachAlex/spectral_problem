"""Equal component weights must not depend on the number of residual points."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_conditions import EqualInitialAdaptiveConditionsLosses


class PointwiseLosses:
    def __init__(self, counts):
        self.counts = counts
        self.values = torch.arange(1.0, len(counts) + 1, requires_grad=True)

    def calculate_mean_losses(self):
        return [self.values[i].square().expand(count).mean()
                for i, count in enumerate(self.counts)]

    def calculate_mean_losses_and_counts(self):
        raise AssertionError("Equal initialization must not request point counts.")


class AdaptiveConditionsTests(unittest.TestCase):
    def test_equal_weights_for_all_three_experiments(self):
        for k in (4, 5, 7):
            with self.subTest(components=k):
                trainer = SimpleNamespace(pinn=PointwiseLosses([100000] + [1] * (k - 1)))
                regularization = EqualInitialAdaptiveConditionsLosses(lr=1e-4)
                total, losses = regularization(trainer)
                torch.testing.assert_close(regularization.adaptive_cond.train_weight,
                                           torch.zeros(k))
                torch.testing.assert_close(regularization.adaptive_cond.weight,
                                           torch.full((k,), 1 / k))
                torch.testing.assert_close(total, losses.mean())

    def test_replicating_points_does_not_change_initial_weights_or_loss(self):
        results = []
        for counts in ([10, 1, 1], [100000, 1, 1]):
            trainer = SimpleNamespace(pinn=PointwiseLosses(counts))
            regularization = EqualInitialAdaptiveConditionsLosses(lr=1e-4)
            total, losses = regularization(trainer)
            results.append((total.detach(), losses, regularization.adaptive_cond.weight.detach()))
        for first, second in zip(*results):
            torch.testing.assert_close(first, second)

    def test_adaptation_survives_point_update(self):
        trainer = SimpleNamespace(pinn=PointwiseLosses([10, 1, 1]), current_epoch=0,
                                  update_grid_every=1000)
        regularization = EqualInitialAdaptiveConditionsLosses(lr=1e-4)
        total, _ = regularization(trainer)
        adaptive = regularization.adaptive_cond
        total.backward()
        # Simulate a resampling boundary; point counts must not reinitialize logits.
        trainer.current_epoch = 1000
        trainer.pinn.counts = [100000, 1, 1]
        regularization(trainer)
        self.assertIs(regularization.adaptive_cond, adaptive)
        self.assertGreater(adaptive.train_weight.detach().abs().max().item(), 0)
        self.assertGreater(adaptive.weight[-1].item(), adaptive.weight[0].item())
        self.assertEqual(adaptive.optimizer.state[adaptive.train_weight]["step"].item(), 1)


if __name__ == "__main__":
    unittest.main()
