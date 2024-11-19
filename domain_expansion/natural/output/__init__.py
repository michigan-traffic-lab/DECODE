# Adapted from natural-posterior-network: https://github.com/borchero/natural-posterior-network.git
# Original Paper: Natural Posterior Network: Deep Bayesian Predictive Uncertainty for Exponential Family Distributions
# Original Authors: Bertrand Charpentier, Oliver Borchert, Daniel Zügner, Simon Geisler, Stephan Günnemann
# Link to the paper: https://arxiv.org/abs/2105.04471
# Licensed under the MIT License: https://opensource.org/licenses/MIT

from ._base import Output
from .categorical import CategoricalOutput

__all__ = ["CategoricalOutput", "Output"]
