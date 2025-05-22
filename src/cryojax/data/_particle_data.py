import abc
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Float

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
    images: eqx.AbstractVar[Float[Array, "... y_dim x_dim"]]


class AbstractParticleParameterDataset(AbstractDataset[T], Generic[T]):
    @abc.abstractmethod
    def append(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loads_metadata(self) -> bool:
        raise NotImplementedError

    @loads_metadata.setter
    @abc.abstractmethod
    def loads_metadata(self, value: bool):
        raise NotImplementedError


class AbstractParticleStackDataset(AbstractDataset[T], Generic[T]):
    @property
    @abc.abstractmethod
    def param_dataset(self) -> AbstractParticleParameterDataset:
        raise NotImplementedError
