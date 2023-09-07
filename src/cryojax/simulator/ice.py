"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "ExpIce", "EmpiricalIce"]

from abc import ABCMeta, abstractmethod
from typing import Optional, Any

from .kernel import Sum, Constant, Exp, Empirical
from .noise import GaussianNoise
from ..core import dataclass, field, Array, ArrayLike, Parameter, CryojaxObject


@dataclass
class Ice(CryojaxObject, metaclass=ABCMeta):
    """
    Base class for an ice model.
    """

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Array:
        """Sample a realization from the ice model."""
        raise NotImplementedError


@dataclass
class NullIce(GaussianNoise, Ice):
    """
    A 'null' ice model.
    """

    def variance(self, freqs: Optional[ArrayLike] = None) -> Array:
        return 0.0


@dataclass
class EmpiricalIce(GaussianNoise, Ice):
    """
    Ice modeled as gaussian noise with a
    measured power spectrum.

    For more detail, see
    ``cryojax.simulator.kernel.Empirical``.

    Attributes
    ----------
    spectrum : `jax.Array`, shape `(N1, N2)`
        The measured power spectrum.
    kappa_i : `cryojax.core.Parameter`
        A scale factor for the variance.
    lambda_i : `cryojax.core.Parameter`
        A variance offset.
    """

    spectrum: Array = field(pytree_node=False)
    kernel: Sum = field(pytree_node=False, init=False, encode=False)

    kappa_i: Parameter = 1.0
    lambda_i: Parameter = 0.0

    def __post_init__(self):
        empirical = Empirical(
            measurement=self.spectrum,
            amplitude=self.kappa_i,
        )
        constant = Constant(value=self.lambda_i)
        object.__setattr__(self, "kernel", Sum(empirical, constant))

    def variance(self, freqs: Optional[ArrayLike] = None) -> Array:
        """Power spectrum measured from a micrograph."""
        return self.kernel(freqs)


@dataclass
class ExpIce(GaussianNoise, Ice):
    r"""
    Ice modeled as gaussian noise with a covariance
    matrix equal to an exponential decay. For more
    detail, see ``cryojax.simulator.kernel.Exp``.

    Attributes
    ----------
    kappa_i : `cryojax.core.Parameter`
        The "coupling strength".
    beta_i : `cryojax.core.Parameter`
        The correlation length.
    lambda_i : `cryojax.core.Parameter`
        The "white" part of the variance.
    """

    kernel: Sum = field(pytree_node=False, init=False, encode=False)

    kappa_i: Parameter = 0.1
    beta_i: Parameter = 1.0
    lambda_i: Parameter = 0.1

    def __post_init__(self):
        exp = Exp(
            amplitude=self.kappa_i,
            beta=self.beta_i,
        )
        constant = Constant(value=self.lambda_i)
        object.__setattr__(self, "kernel", Sum(exp, constant))

    def variance(self, freqs: ArrayLike) -> Array:
        """Power spectrum modeled by a pure exponential."""
        return self.kernel(freqs)
