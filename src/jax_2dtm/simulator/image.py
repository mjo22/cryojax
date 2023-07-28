"""
Routines to model image formation.
"""

from __future__ import annotations

__all__ = [
    "Image",
    "ScatteringImage",
    "OpticsImage",
    "GaussianImage",
]

from abc import ABCMeta, abstractmethod
from typing import Union, Optional
from dataclasses import InitVar

import jax.numpy as jnp

from .cloud import Cloud
from .filters import AntiAliasingFilter
from .noise import GaussianNoise
from .state import ParameterState, ParameterDict
from ..utils import fft, ifft
from ..core import dataclass, field, Array, Scalar
from . import Filter, Mask, ScatteringConfig


@dataclass
class Image(metaclass=ABCMeta):
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
    filters : `list[Filter]`
        A list of filters to apply to the image. By default, this is an
        ``AntiAliasingFilter`` with its default configuration.
    observed : `jax.Array`, optional
        The observed data in Fourier space. This must be the same
        shape as ``config.shape``. ``ImageModel.observed`` will return
        the data with the filters applied.
    """

    state: ParameterState
    config: ScatteringConfig = field(pytree_node=False)
    cloud: Cloud = field(pytree_node=False)

    filters: list[Filter] = field(pytree_node=False, init=False)
    masks: list[Mask] = field(pytree_node=False, init=False)
    observed: Optional[Array] = field(pytree_node=False, init=False)

    filters: InitVar[list[Filter] | None] = None
    masks: InitVar[list[Filter] | None] = None
    observed: InitVar[Array | None] = None

    def __post_init__(self, filters, masks, observed):
        # Set filters
        filters = (
            [
                AntiAliasingFilter(
                    self.config.pixel_size * self.config.padded_freqs
                )
            ]
            if filters is None
            else filters
        )
        object.__setattr__(self, "filters", filters)
        # Set masks
        masks = [] if masks is None else masks
        object.__setattr__(self, "masks", masks)
        # Set observed data
        if observed is not None:
            assert self.config.shape == observed.shape
            observed = self.config.upsample(observed)
            assert self.config.padded_shape == observed.shape
            observed = self.mask(self.config.downsample(self.filter(observed)))
            assert self.config.shape == observed.shape
        object.__setattr__(self, "observed", observed)

    @abstractmethod
    def render(
        self, state: Optional[ParameterState] = None, crop: bool = True
    ) -> Array:
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

    def filter(self, image: Array) -> Array:
        """Apply filters to image."""
        for filter in self.filters:
            image = filter(image)
        return image

    def mask(self, image: Array) -> Array:
        """Apply masks to image."""
        if len(self.masks) > 0:
            image = ifft(image)
            for mask in self.masks:
                image = mask(image)
            image = fft(image)
        return image

    def residuals(self, state: Optional[ParameterState] = None):
        """Return the residuals between the model and observed data."""
        state = state or self.state
        simulated = self.render(state)
        residuals = self.observed - simulated
        return residuals

    def update(self, params: Union[ParameterDict, ParameterState]) -> Image:
        """Return a new ImageModel based on a new ParameterState."""
        state = self.state.update(params) if type(params) is dict else params
        return self.replace(state=state)


@dataclass
class ScatteringImage(Image):
    """
    Compute the scattering pattern on the imaging plane.
    """

    def render(
        self, state: Optional[ParameterState] = None, crop: bool = True
    ) -> Array:
        """Render the scattering pattern"""
        state = state or self.state
        # Compute scattering at image plane.
        cloud = self.cloud.view(state.pose)
        scattering_image = cloud.project(self.config)
        # Apply filters
        scattering_image = self.filter(scattering_image)
        # Crop
        if crop:
            scattering_image = self.config.crop(scattering_image)

        return scattering_image

    def sample(self, state: Optional[ParameterState] = None) -> Array:
        state = state or self.state
        simulated = self.render(state)
        return self.config.crop(simulated)

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
        scattering_image = super().render(state, crop=False)
        # Compute and apply CTF
        ctf = state.optics(self.config.padded_freqs)
        optics_image = ctf * scattering_image
        # Crop
        if self.config.pad_scale != 1:
            optics_image = self.config.crop(optics_image)
        # Apply masks to image
        masked_image = self.mask(optics_image)
        # Rescale the image to desired mean and standard deviation
        rescaled_image = state.intensity.rescale(masked_image)

        return rescaled_image


@dataclass
class GaussianImage(OpticsImage):
    """
    Sample an image from a gaussian noise model, or compute
    the log-likelihood.

    Note that this computes the likelihood in Fourier space,
    which allows one to model an arbitrary noise power spectrum.
    """

    def __post_init__(self, *args):
        super().__post_init__(*args)
        assert isinstance(self.state.noise, GaussianNoise)

    def sample(self, state: Optional[ParameterState] = None) -> Array:
        """Sample an image from a realization of the noise"""
        state = state or self.state
        simulated = self.render(state)
        noise = state.noise.sample(self.config.freqs * self.config.pixel_size)
        return simulated + self.mask(noise)

    def log_likelihood(self, state: Optional[ParameterState] = None) -> Scalar:
        """Evaluate the log-likelihood of the data given a parameter set."""
        state = state or self.state
        residuals = self.residuals(state)
        variance = state.noise.variance(
            self.config.freqs * self.config.pixel_size
        )
        loss = jnp.sum((residuals * jnp.conjugate(residuals)) / (2 * variance))
        return loss.real / residuals.size
