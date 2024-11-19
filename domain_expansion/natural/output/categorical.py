import torch
from torch import nn
import domain_expansion.natural.distributions as D
from ._base import Output


class CategoricalOutput(Output):
    """
    Categorical output with uniformative Dirichlet prior.
    """

    def __init__(self, num_classes: int):
        """
        Args:
            dim: The dimension of the latent space.
            num_classes: The number of categories for the output distribution.
        """
        super().__init__()
        self.prior = D.DirichletPrior(num_categories=num_classes, evidence=num_classes)

    def forward(self, x: torch.Tensor) -> D.Likelihood:
        return D.Categorical(x.log_softmax(-1))
