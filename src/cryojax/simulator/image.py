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

from .filters import AntiAliasingFilter
from .noise import GaussianNoise
from .state import ParameterState, ParameterDict
from .intensity import rescale_image
from ..utils import fft, ifft
from ..core import dataclass, field, Array, Scalar
from . import Filter, Mask, Specimen, ScatteringConfig


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
    scattering : `jax_2dtm.simulator.ScatteringConfig`
        The image and scattering model configuration.
    specimen : `jax_2dtm.simulator.Specimen`
        The specimen from which to render images.
    state : `jax_2dtm.simulator.ParameterState`
        The parameter state of the model.
    filters : `list[Filter]`
        A list of filters to apply to the image. By default, this is an
        ``AntiAliasingFilter`` with its default configuration.
    observed : `jax.Array`, optional
        The observed data in Fourier space. This must be the same
        shape as ``scattering.shape``. ``ImageModel.observed`` will return
        the data with the filters applied.
    """

    state: ParameterState
    scattering: ScatteringConfig = field(pytree_node=False)
    specimen: Specimen = field(pytree_node=False)

    filters: list[Filter] = field(pytree_node=False, init=False)
    masks: list[Mask] = field(pytree_node=False, init=False)
    observed: Optional[Array] = field(pytree_node=False, init=False)

    filters: InitVar[list[Filter] | None] = None
    masks: InitVar[list[Filter] | None] = None
    observed: InitVar[Array | None] = None
    _process_observed: bool = field(pytree_node=False, default=True)

    def __post_init__(self, filters, masks, observed):
        # Set filters
        filters = (
            [
                AntiAliasingFilter(
                    self.scattering.pixel_size * self.scattering.padded_freqs
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
        if observed is not None and self._process_observed:
            assert self.scattering.shape == observed.shape
            observed = ifft(observed)
            mean, std = observed.mean(), observed.std()
            observed = fft(self.scattering.pad(observed, constant_values=mean))
            assert self.scattering.padded_shape == observed.shape
            observed = self.scattering.crop(ifft(self.filter(observed)))
            assert self.scattering.shape == observed.shape
            observed = rescale_image(observed, std, mean)
            observed = fft(self.mask(observed))
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
        for mask in self.masks:
            image = mask(image)
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
        return self.replace(state=state, _process_observed=False)


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
        specimen = self.specimen.view(state.pose)
        scattering_image = specimen.scatter(self.scattering)
        # Apply filters
        scattering_image = self.filter(scattering_image)
        # Optional cropping to view image at this stage in the pipeline
        if crop:
            scattering_image = fft(
                self.mask(self.scattering.crop(ifft(scattering_image)))
            )

        return scattering_image

    def sample(self, state: Optional[ParameterState] = None) -> Array:
        state = state or self.state
        simulated = self.render(state)
        return self.scattering.crop(simulated)

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
        # Compute scattering at object plane.
        scattering_image = super().render(state, crop=False)
        # Compute and apply CTF
        ctf = state.optics(self.scattering.padded_freqs)
        optics_image = ctf * scattering_image
        # Crop and rescale
        rescaled_image = state.intensity.rescale(
            self.scattering.crop(ifft(optics_image))
        )
        # Mask the image
        masked_image = fft(self.mask(rescaled_image))

        return masked_image


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
        noise = state.noise.sample(
            self.scattering.freqs * self.scattering.pixel_size
        )
        return simulated + fft(self.mask(ifft(noise)))

    def log_likelihood(self, state: Optional[ParameterState] = None) -> Scalar:
        """Evaluate the log-likelihood of the data given a parameter set."""
        state = state or self.state
        residuals = self.residuals(state)
        variance = state.noise.variance(
            self.scattering.freqs * self.scattering.pixel_size
        )
        loss = jnp.sum((residuals * jnp.conjugate(residuals)) / (2 * variance))
        return loss.real / residuals.size
