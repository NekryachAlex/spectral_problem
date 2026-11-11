"""MultiPINN AdaptiveConditions with equal initial weights per loss component."""
from multipinn.regularization import AdaptiveConditionsLosses
from multipinn.regularization.adaptive import AdaptiveWeightSum


class EqualInitialAdaptiveConditionsLosses(AdaptiveConditionsLosses):
    """Change initialization only; retain MultiPINN's adaptive update mechanism."""

    def __call__(self, trainer):
        if self.adaptive_cond is None:
            mean_losses = trainer.pinn.calculate_mean_losses()
            # log(1) = 0, so softmax starts at 1/K, regardless of point counts.
            self.adaptive_cond = AdaptiveWeightSum([1.0] * len(mean_losses), self.lr)
            return self.weight_sum(mean_losses)
        return super().__call__(trainer)
