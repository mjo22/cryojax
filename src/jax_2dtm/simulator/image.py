"""
Routines to model image formation.
"""

from __future__ import annotations

__all__ = [
    "ImageConfig",
    "ImageModel",
    "ScatteringImage",
    "OpticsImage",
    "GaussianImage",
]

import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Union, Optional, Any

import jax
import jax.numpy as jnp
from functools import partial

from .scattering import ScatteringConfig
from .cloud import Cloud
from .filters import Filter, AntiAliasingFilter
from .noise import GaussianNoise
from .state import ParameterState, ParameterDict
from ..types import Array, Scalar
from ..utils import fftfreqs, fft


@dataclasses.dataclass
class ImageModel(metaclass=ABCMeta):
    """
    Base class for an imaging model.

    Note that only the ``ImageModel.state`` field
    is mutable.

    Attributes
    ----------
    config : `jax_2dtm.simulator.ScatteringConfig`
        The image and scattering model configuration.
    cloud : `jax_2dtm.simulator.Cloud`
        The point cloud used to render images.
    state : `jax_2dtm.simulator.ParameterState`
        The parameter state of the model.
    freqs : `jax.Array`
        The fourier wave vectors in the imaging plane.
    observed : `jax.Array`, optional
        The observed data in real space. This must be the same
        shape as ``config.shape``. To ensure there
        are no mistakes in Fourier convention, ``ImageModel.observed``
        returns the observed data in Fourier space.
    """

    config: ScatteringConfig
    cloud: Cloud
    state: ParameterState
    freqs: Array = dataclasses.field(init=False)
    observed: Optional[Array] = dataclasses.field(init=False)

    def __post_init__(self, observed: Optional[Array] = None):
        # Set additional fields and check arguments.
        self.freqs = fftfreqs(self.config.shape, self.config.pixel_size)
        if observed is not None:
            assert all(self.config.shape == observed.shape)
            self.observed = fft(observed)
        else:
            self.observed = None

    @abstractmethod
    def render(self, state: ParameterState) -> Array:
        """Render an image given a parameter set."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: ParameterState) -> Array:
        """Sample the an image from a realization of the noise"""
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, observed: Array, state: ParameterState) -> Scalar:
        """Evaluate the log-likelihood of the data given a parameter set."""
        raise NotImplementedError

    def __call__(
        self,
        params: ParameterDict = {},
    ) -> Union[Array, Scalar]:
        """
        Evaluate the model at a parameter set.

        If ``ImageModel.observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        self.state = self.state.update(params)
        if self.observed is None:
            return self.sample(self.state)
        else:
            return self.log_likelihood(self.observed, self.state)

    @property
    def _mutable(self):
        """Define which fields are mutable."""
        return ["state"]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self._mutable or not hasattr(self, __name):
            super().__setattr__(__name, __value)
        else:
            raise TypeError(f"Attribute '{__name}' is immutable!")


@dataclasses.dataclass
class ScatteringImage(ImageModel):
    """
    Compute the scattering pattern on the imaging plane.

    Attributes
    ----------
    filters : `list[Filter]`
        A list of filters to apply to the image. This
        field is mutable (see ``ScatteringImage.filters``)
        for details. By default, this is an
        ``AntiAliasingFilter`` with its default configuration.
    """

    filters: list[Filter] = dataclasses.field(init=False)

    def __post_init__(self, filters: Optional[list[Filter]] = None):
        super().__post_init__()
        self.filters = filters or [AntiAliasingFilter(self.config, self.freqs)]
        assert all([filter.config is self.config for filter in self.filters])
        assert all([filter.freqs is self.freqs for filter in self.filters])

    def render(self, state: ParameterState) -> Array:
        """Render the scattering pattern"""
        # Compute scattering at image plane.
        cloud = self.cloud.view(state.pose)
        scattering_image = cloud.project(self.config)
        # Apply filters
        for filter in self.filters:
            scattering_image = filter(scattering_image)

        return scattering_image

    def sample(self, state: ParameterState) -> Array:
        return self.render(state)

    def log_likelihood(self, observed: Array, state: ParameterState) -> Scalar:
        raise NotImplementedError


@dataclasses.dataclass
class OpticsImage(ScatteringImage):
    """
    Compute the scattering pattern on the imaging plane,
    moduated by a CTF.
    """

    def render(self, state: ParameterState) -> Array:
        """Render an image from a model of the CTF."""
        # Compute scattering at image plane.
        scattering_image = super().render(state)
        # Compute and apply CTF
        ctf = state.optics(self.freqs)
        optics_image = ctf * scattering_image

        return optics_image


@dataclasses.dataclass
class GaussianImage(OpticsImage):
    """
    Sample an image from a gaussian noise model, or compute
    the log-likelihood.

    Note that this computes the likelihood in Fourier space,
    which allows for modeling of an arbitrary noise power spectrum.
    """

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.state.noise, GaussianNoise)

    def sample(self, state: ParameterState) -> Array:
        """Sample an image from a realization of the noise"""
        return self.render(state) + state.noise.sample(self.config, self.freqs)

    def log_likelihood(self, observed: Array, state: ParameterState) -> Scalar:
        """Evaluate the log-likelihood of the data given a parameter set."""
        simulated = self.render(state)
        residual = observed - simulated
        variance = state.noise.variance(self.freqs)
        loss = jnp.sum(
            (residual * jnp.conjugate(residual)).real / (2 * variance)
        )
        return loss
