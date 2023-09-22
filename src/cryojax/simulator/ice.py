"""
Abstraction of the ice in a cryo-EM image.
"""

__all__ = ["Ice", "NullIce", "GaussianIce"]

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import jax.numpy as jnp

from .scattering import ScatteringConfig
from .kernel import Kernel, Exp
from .optics import Optics
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

    @abstractmethod
    def scatter(
        self,
        scattering: ScatteringConfig,
        resolution: float,
        optics: Optional[Optics] = None,
        **kwargs: Any,
    ) -> Array:
        """
        Scatter a realization of ice model onto the imaging plane.

        Arguments
        ---------
        scattering : `cryojax.simulator.ScatteringConfig`
            The scattering configuration.
        resolution : `float`
            The resolution of the image.
        optics : `cryojax.simulator.Optics`, optional
            The instrument optics.
        """
        raise NotImplementedError


@dataclass
class NullIce(Ice):
    """
    A 'null' ice model.
    """

    def sample(self, freqs: ArrayLike) -> Array:
        return jnp.zeros(jnp.asarray(freqs).shape[0:-1])

    def scatter(
        self,
        scattering: ScatteringConfig,
        resolution: float,
        optics: Optional[Optics] = None,
        **kwargs: Any,
    ) -> Array:
        freqs = scattering.padded_freqs / resolution
        return self.sample(freqs, **kwargs)


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

    def scatter(
        self,
        scattering: ScatteringConfig,
        resolution: float,
        optics: Optional[Optics] = None,
        **kwargs: Any,
    ) -> Array:
        # Sample an ice realization
        freqs = scattering.padded_freqs / resolution
        ice = self.sample(freqs, **kwargs)
        # Compute and apply CTF
        if optics is not None:
            ctf = optics(freqs)
            ice = optics.apply(ctf, ice)

        return ice
