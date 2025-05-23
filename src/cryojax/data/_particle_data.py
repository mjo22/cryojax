import abc
import pathlib
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Inexact

from ..internal import NDArrayLike
from ..simulator import (
    AbstractPose,
    AbstractTransferTheory,
    InstrumentConfig,
)
from ._dataset import AbstractDataset


T = TypeVar("T")


class AbstractParticleParameters(eqx.Module):
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


class AbstractParticleParameterDataset(AbstractDataset[T], Generic[T]):
    @abc.abstractmethod
    def __setitem__(self, index, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError


class AbstractParticleStackDataset(AbstractDataset[T], Generic[T]):
    @abc.abstractmethod
    def __setitem__(self, index, value: T | Inexact[NDArrayLike, "... y_dim x_dim"]):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def write_image_stack(
        self,
        path_to_output: str | pathlib.Path,
        image_stack: Inexact[NDArrayLike, "... y_dim x_dim"],
    ):
        raise NotImplementedError
