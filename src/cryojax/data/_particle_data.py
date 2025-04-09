import abc
from typing import Generic, Optional, TypeVar

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

    metadata: eqx.AbstractVar[Optional[dict]]


class AbstractParticleStack(eqx.Module):
    """Images from a particle stack, along with information
    of their parameters.
    """

    parameters: eqx.AbstractVar[AbstractParticleParameters]
    images: eqx.AbstractVar[Float[Array, "... y_dim x_dim"]]


class AbstractParticleParameterReader(AbstractDataset[T], Generic[T]):
    @property
    @abc.abstractmethod
    def loads_metadata(self) -> bool:
        raise NotImplementedError

    @loads_metadata.setter
    @abc.abstractmethod
    def loads_metadata(self, value: bool):
        raise NotImplementedError


class AbstractParticleStackReader(AbstractDataset[T], Generic[T]):
    @property
    @abc.abstractmethod
    def param_reader(self) -> AbstractParticleParameterReader:
        raise NotImplementedError
