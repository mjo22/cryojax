"""
Routines to model image formation.
"""

__all__ = ["ImageConfig", "ImageModel"]


import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Union, Optional
from ..types import dataclass, field, Array, Scalar
from .state import ParameterState
from .cloud import Cloud


@dataclass
class ImageConfig:
    """
    Attributes
    ----------
    shape : tuple[int, int, int]
        Shape of the imaging volume in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pixel_size : float
        Size of camera pixels, in dimensions of length.
    eps : float
        Desired precision in computing the volume
        projection. See `finufft <https://finufft.readthedocs.io/en/latest/>`_
        for more detail.
    """

    shape: tuple[int, int] = field(pytree_node=False)
    pixel_size: float = field(pytree_node=False)
    eps: float = field(pytree_node=False, default=1e-6)


@dataclasses.dataclass
class ImageModel(metaclass=ABCMeta):
    """Base class for an imaging model."""

    config: ImageConfig
    cloud: Cloud
    observed: Optional[Array] = None

    @abstractmethod
    def render(self, params: ParameterState) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample(self, params: ParameterState) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(
        self, observed: Array, params: ParameterState
    ) -> Scalar:
        raise NotImplementedError

    def __call__(
        self,
        params: ParameterState,
    ) -> Union[Array, Scalar]:
        if self.observed is None:
            return self.render(params)
        else:
            return self.log_likelihood(self.observed, params)
