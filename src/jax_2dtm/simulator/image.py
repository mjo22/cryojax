"""
Routines to model image formation.
"""

from __future__ import annotations

__all__ = ["ImageConfig", "ImageModel", "ScatteringImage", "OpticsImage"]

import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Union, Optional

from .scattering import ScatteringConfig
from .cloud import Cloud
from .state import ParameterState, ParameterDict
from .filters import Filter, AntiAliasingFilter
from ..types import Array, Scalar
from ..utils import fftfreqs


@dataclasses.dataclass
class ImageModel(metaclass=ABCMeta):
    """
    Base class for an imaging model.

    Attributes
    ----------
    config : `jax_2dtm.simulator.ScatteringConfig`
    cloud : `jax_2dtm.simulator.Cloud`
    state : `jax_2dtm.simulator.State`
    """

    config: ScatteringConfig
    cloud: Cloud
    state: Optional["ParameterState"] = None
    observed: Optional[Array] = None

    def __post_init__(self):
        self.freqs: Array = fftfreqs(self.config.shape, self.config.pixel_size)

    @abstractmethod
    def render(self, state: "ParameterState") -> Array:
        """Render an image given a parameter set."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: "ParameterState") -> Array:
        """Sample the an image from a realization of the noise"""
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(
        self, observed: Array, state: "ParameterState"
    ) -> Scalar:
        """Evaluate the log-likelihood of the data given a parameter set."""
        raise NotImplementedError

    def __call__(
        self,
        params: Union["ParameterState", "ParameterDict"],
    ) -> Union[Array, Scalar]:
        """
        Evaluate the model at a parameter set.

        If ``ImageModel.observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        If there is no noise model, render an image.
        """
        self.state = (
            self.state.update(params) if type(params) is dict else params
        )
        if self.observed is None:
            return self.render(self.state)
        else:
            return self.log_likelihood(self.observed, self.state)


@dataclasses.dataclass
class ScatteringImage(ImageModel):
    """
    Compute the scattering pattern on the imaging plane.
    """

    def __post_init__(self):
        super().__post_init__()
        self.filters: list[Filter] = [
            AntiAliasingFilter(self.config, self.freqs)
        ]

    def render(self, state: "ParameterState") -> Array:
        # Compute scattering at image plane
        cloud = self.cloud.view(state.pose)
        scattering_image = cloud.project(self.config)
        # Apply filters
        for filter in self.filters:
            scattering_image = filter(scattering_image)

        return scattering_image

    def sample(self, state: "ParameterState") -> Array:
        raise NotImplementedError

    def log_likelihood(
        self, observed: Array, state: "ParameterState"
    ) -> Scalar:
        raise NotImplementedError


@dataclasses.dataclass
class OpticsImage(ScatteringImage):
    """
    Compute the scattering pattern on the imaging plane,
    moduated by a CTF.
    """

    def render(self, state: "ParameterState") -> Array:
        """
        Render an image from a model of the CTF.
        """
        # Compute scattering at image plane.
        cloud = self.cloud.view(state.pose)
        scattering_image = cloud.project(self.config)
        # Compute and apply CTF
        ctf = state.optics(self.freqs)
        optics_image = ctf * scattering_image
        # Apply filters
        for filter in self.filters:
            optics_image = filter(optics_image)

        return optics_image
