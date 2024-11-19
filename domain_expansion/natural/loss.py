# Adapted from natural-posterior-network: https://github.com/borchero/natural-posterior-network.git
# Original Paper: Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions
# Original Authors: Bertrand Charpentier, Oliver Borchert, Daniel Zügner, Simon Geisler, Stephan Günnemann
# Link to the paper: https://arxiv.org/abs/2105.04471
# Licensed under the MIT License: https://opensource.org/licenses/MIT

from typing import Literal
import torch
from torch import nn
from domain_expansion.natural.distributions import Posterior

Reduction = Literal["mean", "sum", "none"]


class BayesianLoss(nn.Module):
    """
    The Bayesian loss computes an uncertainty-aware loss based on the parameters of a conjugate
    prior of the target distribution.
    """

    def __init__(self, entropy_weight: float = 0.0, reduction: Reduction = "mean"):
        """
        Args:
            entropy_weight: The weight for the entropy regualarizer.
            reduction: The reduction to apply to the loss. Must be one of "mean", "sum", "none".
        """
        super().__init__()
        self.entropy_weight = entropy_weight
        self.reduction = reduction

    def forward(self, y_pred: Posterior, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss of the prediction with respect to the target.

        Args:
            y_pred: The posterior distribution predicted by the Natural Posterior Network.
            y_true: The true targets. Either indices for classes of a classification problem or
                the real target values. Must have the same batch shape as ``y_pred``.

        Returns:
            The loss, processed according to ``self.reduction``.
        """
        nll = -y_pred.expected_log_likelihood(y_true)
        loss = nll - self.entropy_weight * y_pred.entropy()

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
