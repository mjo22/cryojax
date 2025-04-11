import abc
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
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


class ParticleStack(eqx.Module, strict=True):
    """Images from a particle stack, along with information
    of their parameters.
    """

    parameters: AbstractParticleParameters
    images: Float[Array, "... y_dim x_dim"]

    def __init__(
        self,
        parameters: AbstractParticleParameters,
        images: Float[Array, "... y_dim x_dim"],
    ):
        """**Arguments:**

        - `parameters`:
            The image parameters, represented as
            an `AbstractParticleParameters` object.
        - `images`:
            The stack of images. The shape of this array
            is a leading batch dimension followed by the shape
            of an image in the stack.
        """
        # Set the image parameters
        self.parameters = parameters
        # Set the image stack
        self.images = jnp.asarray(images)


class AbstractParticleParameterDataset(AbstractDataset[T], Generic[T]):
    @property
    @abc.abstractmethod
    def loads_metadata(self) -> bool:
        raise NotImplementedError

    @loads_metadata.setter
    @abc.abstractmethod
    def loads_metadata(self, value: bool):
        raise NotImplementedError


class AbstractParticleStackDataset(AbstractDataset[ParticleStack]):
    @property
    @abc.abstractmethod
    def param_dataset(self) -> AbstractParticleParameterDataset:
        raise NotImplementedError
