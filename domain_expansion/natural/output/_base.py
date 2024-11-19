# Adapted from natural-posterior-network: https://github.com/borchero/natural-posterior-network.git
# Original Paper: Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions
# Original Authors: Bertrand Charpentier, Oliver Borchert, Daniel Zügner, Simon Geisler, Stephan Günnemann
# Link to the paper: https://arxiv.org/abs/2105.04471
# Licensed under the MIT License: https://opensource.org/licenses/MIT

from abc import ABC, abstractmethod
import torch
from torch import nn
import domain_expansion.natural.distributions as D


class Output(nn.Module, ABC):
    """
    Base class for output distributions of NatPN.
    """

    prior: D.ConjugatePrior

    @abstractmethod
    def forward(self, x: torch.Tensor) -> D.Likelihood:
        """
        Derives the likelihood distribution from the latent representation via a linear mapping
        to the distribution parameters.

        Args:
            x: The inputs' latent representations.

        Returns:
            The distribution.
        """
