import abc
import pathlib
from typing import Any, Generic, Literal, Optional, TypeVar

import equinox as eqx
import numpy as np
from jaxtyping import Array, Float, Inexact, Int

from ...internal import NDArrayLike
from ...simulator import (
    AbstractPose,
    AbstractTransferTheory,
    InstrumentConfig,
)
from .._dataset import AbstractDataset


T = TypeVar("T")


class AbstractParticleParameters(eqx.Module, strict=True):
    """Parameters for a particle stack."""

    instrument_config: eqx.AbstractVar[InstrumentConfig]
    pose: eqx.AbstractVar[AbstractPose]
    transfer_theory: eqx.AbstractVar[AbstractTransferTheory]


class AbstractParticleStack(eqx.Module, strict=True):
    """Images from a particle stack, along with information
    of their parameters.
    """

    parameters: eqx.AbstractVar[AbstractParticleParameters]
    images: eqx.AbstractVar[Inexact[Array, "... y_dim x_dim"]]


class AbstractParticleParameterFile(AbstractDataset[T], Generic[T]):
    @abc.abstractmethod
    def __setitem__(self, index, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path_to_output(self) -> pathlib.Path:
        raise NotImplementedError

    @path_to_output.setter
    @abc.abstractmethod
    def path_to_output(self, value: str | pathlib.Path):
        raise NotImplementedError

    @property
    def mode(self) -> Literal["r", "w"]:
        raise NotImplementedError


class AbstractParticleStackDataset(AbstractDataset[T], Generic[T]):
    @property
    @abc.abstractmethod
    def parameter_file(self) -> AbstractParticleParameterFile:
        raise NotImplementedError

    @abc.abstractmethod
    def __setitem__(self, index, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def write_images(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: Optional[Any] = None,
    ):
        raise NotImplementedError

    @property
    def mode(self) -> Literal["r", "w"]:
        raise NotImplementedError
