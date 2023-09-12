"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "GaussianIce"]

from abc import ABCMeta, abstractmethod
from typing import Any

import jax.numpy as jnp

from .kernel import Kernel, Exp
from .noise import GaussianNoise
from ..core import dataclass, field, Array, ArrayLike, CryojaxObject


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
class NullIce(Ice):
    """
    A 'null' ice model.
    """

    def sample(self, freqs: ArrayLike) -> Array:
        return jnp.zeros(jnp.asarray(freqs).shape[0:-1])


@dataclass
class GaussianIce(GaussianNoise, Ice):
    r"""
    Ice modeled as gaussian noise.

    Attributes
    ----------
    variance : `cryojax.simulator.Kernel`
        A kernel that computes the variance
        of the ice, modeled as noise. By default,
        ``Exp()``.
    """

    variance: Kernel = field(default_factory=Exp)
