from typing import Optional, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..simulator import (
    AbstractPose,
    ContrastTransferTheory,
    InstrumentConfig,
)
from ._dataset import AbstractDataset


T = TypeVar("T")


class ParticleParameters(eqx.Module):
    """Parameters for a particle stack."""

    instrument_config: InstrumentConfig
    pose: AbstractPose
    transfer_theory: ContrastTransferTheory

    particle_data: Optional[dict] = eqx.field(static=True)

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        pose: AbstractPose,
        transfer_theory: ContrastTransferTheory,
        *,
        particle_data: Optional[dict] = None,
    ):
        """**Arguments:**

        - `instrument_config`:
            The instrument configuration. Any subset of pytree leaves may
            have a batch dimension.
        - `pose`:
            The pose, represented by euler angles. Any subset of pytree leaves may
            have a batch dimension.
        - `transfer_theory`:
            The contrast transfer theory. Any subset of pytree leaves may
            have a batch dimension.
        - `particle_dataframe`:
            The raw particle metadata as a `pandas` dataframe
        """
        # Set instrument config as is
        self.instrument_config = instrument_config
        # Set CTF using the defocus offset in the EulerAnglePose
        self.transfer_theory = transfer_theory
        # Set defocus offset to zero
        self.pose = pose
        # Optionally, store the raw metadata
        self.particle_data = particle_data


class ParticleImages(eqx.Module):
    """Images from a particle stack, along with information
    of their parameters.
    """

    parameters: ParticleParameters
    images: Float[Array, "... y_dim x_dim"]

    def __init__(
        self,
        parameters: ParticleParameters,
        images: Float[Array, "... y_dim x_dim"],
    ):
        """**Arguments:**

        - `parameters`:
            The image parameters, represented as
            a `ParticleParameters` object.
        - `images`:
            The stack of images. The shape of this array
            is a leading batch dimension followed by the shape
            of an image in the stack.
        """
        # Set the image parameters
        self.parameters = parameters
        # Set the image stack
        self.images = jnp.asarray(images)


class AbstractParticleParameterReader(
    AbstractDataset[ParticleParameters], strict=True
): ...


class AbstractParticleImageReader(AbstractDataset[ParticleImages], strict=True):
    metadata = eqx.AbstractVar[AbstractParticleParameterReader]
