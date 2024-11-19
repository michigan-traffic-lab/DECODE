from ._base import ConjugatePrior, Likelihood, Posterior, PosteriorPredictive, PosteriorUpdate
from ._utils import mixture_posterior_update
from .categorical import Categorical, DirichletPrior
'''
copied from natural posterior network at "https://github.com/borchero/natural-posterior-network/tree/c1174b8d6d484179d42724dfe20b513fe5569aee"
'''
__all__ = [
    "Categorical",
    "ConjugatePrior",
    "DirichletPrior",
    "Likelihood",
    "Posterior",
    "PosteriorPredictive",
    "PosteriorUpdate",
    "mixture_posterior_update",
]
