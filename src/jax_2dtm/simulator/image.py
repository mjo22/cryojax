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

from abc import ABCMeta, abstractmethod
from typing import Union, Optional
from dataclasses import InitVar

import jax.numpy as jnp

from .scattering import ScatteringConfig
from .cloud import Cloud
from .filters import Filter, AntiAliasingFilter
from .noise import GaussianNoise
from .state import ParameterState, ParameterDict
from ..types import dataclass, field, Array, Scalar
from ..utils import fftfreqs, fft


@dataclass
class ImageModel(metaclass=ABCMeta):
    """
    Base class for an imaging model. Note that the
    model is a PyTree and is therefore immmutable.

    Use ``ImageModel.update`` to return a new model
    with modified parameters, and call ``ImageModel``
    to evaluate the model.

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
    filters : `list[Filter]`
        A list of filters to apply to the image. By default, this is an
        ``AntiAliasingFilter`` with its default configuration.
    observed : `jax.Array`, optional
        The observed data in real space. This must be the same
        shape as ``config.shape``. To ensure there
        are no mistakes in Fourier convention, ``ImageModel.observed``
        returns the observed data in Fourier space.
    """

    state: ParameterState
    config: ScatteringConfig = field(pytree_node=False)
    cloud: Cloud = field(pytree_node=False)

    freqs: Array = field(pytree_node=False, init=False)
    filters: list[Filter] = field(pytree_node=False, init=False)
    observed: Optional[Array] = field(pytree_node=False, init=False)

    filters: InitVar[list[Filter] | None] = None
    observed: InitVar[Array | None] = None

    def __post_init__(self, filters, observed):
        # Set image coordinates
        object.__setattr__(
            self, "freqs", fftfreqs(self.config.shape, self.config.pixel_size)
        )
        # Set filters
        object.__setattr__(
            self,
            "filters",
            filters or [AntiAliasingFilter(self.config, self.freqs)],
        )
        assert all([filter.config is self.config for filter in self.filters])
        assert all(
            [jnp.allclose(filter.freqs, self.freqs) for filter in self.filters]
        )
        # Set observed data
        if observed is not None:
            assert self.config.shape == observed.shape
            observed = fft(observed)
            for filter in self.filters:
                observed = filter(observed)
        object.__setattr__(self, "observed", observed)

    @abstractmethod
    def render(self, state: Optional[ParameterState] = None) -> Array:
        """Render an image given a parameter set."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: Optional[ParameterState] = None) -> Array:
        """Sample the an image from a realization of the noise"""
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, state: Optional[ParameterState] = None) -> Scalar:
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
        state = self.state.update(params)
        if self.observed is None:
            return self.sample(state)
        else:
            return self.log_likelihood(state)

    def residuals(self, state: Optional[ParameterState] = None):
        """Return the residuals between the model and observed data."""
        state = state or self.state
        simulated = self.render(state)
        residuals = self.observed - simulated
        return residuals

    def update(
        self, params: Union[ParameterDict, ParameterState]
    ) -> ImageModel:
        """Return a new ImageModel based on a new ParameterState."""
        state = self.state.update(params) if type(params) is dict else params
        return self.replace(state=state)


@dataclass
class ScatteringImage(ImageModel):
    """
    Compute the scattering pattern on the imaging plane.
    """

    def render(self, state: Optional[ParameterState] = None) -> Array:
        """Render the scattering pattern"""
        state = state or self.state
        # Compute scattering at image plane.
        cloud = self.cloud.view(state.pose)
        scattering_image = cloud.project(self.config)
        # Apply filters
        for filter in self.filters:
            scattering_image = filter(scattering_image)

        return scattering_image

    def sample(self, state: Optional[ParameterState] = None) -> Array:
        state = state or self.state
        return self.render(state)

    def log_likelihood(self, state: Optional[ParameterState] = None) -> Scalar:
        raise NotImplementedError


@dataclass
class OpticsImage(ScatteringImage):
    """
    Compute the scattering pattern on the imaging plane,
    moduated by a CTF and rescaled.
    """

    def render(self, state: Optional[ParameterState] = None) -> Array:
        """Render an image from a model of the CTF."""
        state = state or self.state
        # Compute scattering at image plane.
        scattering_image = super().render(state)
        # Compute and apply CTF
        ctf = state.optics(self.freqs)
        optics_image = ctf * scattering_image
        # Rescale the image to desired scaling and offset
        rescaled_image = state.intensity.rescale(optics_image)

        return rescaled_image


@dataclass
class GaussianImage(OpticsImage):
    """
    Sample an image from a gaussian noise model, or compute
    the log-likelihood.

    Note that this computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.
    """

    def __post_init__(self, filters, observed):
        super().__post_init__(filters, observed)
        assert isinstance(self.state.noise, GaussianNoise)

    def sample(self, state: Optional[ParameterState] = None) -> Array:
        """Sample an image from a realization of the noise"""
        state = state or self.state
        return self.render(state) + state.noise.sample(self.freqs, self.config)

    def log_likelihood(self, state: Optional[ParameterState] = None) -> Scalar:
        """Evaluate the log-likelihood of the data given a parameter set."""
        state = state or self.state
        residuals = self.residuals(state)
        variance = state.noise.variance(self.freqs, self.config)
        loss = jnp.sum((residuals * jnp.conjugate(residuals)) / (2 * variance))
        return loss.real / residuals.size
