"""
Routines to model image formation.
"""

from __future__ import annotations

__all__ = ["ImageConfig", "ImageModel"]

import dataclasses
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Union, Optional

from ..types import dataclass, field, Array, Scalar
from ..utils import fftfreqs

if TYPE_CHECKING:
    from .state import ParameterState, ParameterDict


@dataclass
class ImageConfig:
    """
    Attributes
    ----------
    shape : tuple[int, int]
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pixel_size : float
        Size of camera pixels, in dimensions of length.
    eps : float
        Desired precision in computing the volume
        projection. See `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.
    freqs :
    """

    shape: tuple[int, int] = field(pytree_node=False)
    pixel_size: float = field(pytree_node=False)
    eps: float = field(pytree_node=False, default=1e-6)


@dataclasses.dataclass
class ImageModel(metaclass=ABCMeta):
    """
    Base class for an imaging model.

    Attributes
    ----------

    """

    config: ImageConfig
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
