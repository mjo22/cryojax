from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...simulator import ContrastTransferTheory, EulerAnglePose, InstrumentConfig
from .._particle_data import AbstractParticleParameters, AbstractParticleStack


class RelionParticleParameters(AbstractParticleParameters):
    """Parameters for a particle stack from RELION."""

    instrument_config: InstrumentConfig
    pose: EulerAnglePose
    transfer_theory: ContrastTransferTheory

    metadata: Optional[dict]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        pose: EulerAnglePose,
        transfer_theory: ContrastTransferTheory,
        *,
        metadata: Optional[dict] = None,
    ):
        """**Arguments:**

        - `instrument_config`:
            The instrument configuration.
        - `pose`:
            The pose, represented by euler angles.
        - `transfer_theory`:
            The contrast transfer theory.
        - `metadata`:
            The raw particle metadata as a dictionary.
        """
        # Set instrument config as is
        self.instrument_config = instrument_config
        # Set CTF using the defocus offset in the EulerAnglePose
        self.transfer_theory = transfer_theory
        # Set defocus offset to zero
        self.pose = pose
        # Optionally, store the raw metadata
        self.metadata = metadata


class RelionParticleStack(AbstractParticleStack):
    """Images from a particle stack in RELION, along with information
    of their parameters.
    """

    parameters: RelionParticleParameters
    images: Float[Array, "... y_dim x_dim"]

    def __init__(
        self,
        parameters: RelionParticleParameters,
        images: Float[Array, "... y_dim x_dim"],
    ):
        """**Arguments:**

        - `parameters`:
            The image parameters, represented as
            an `RelionParticleParameters` object.
        - `images`:
            The stack of images. The shape of this array
            is a leading batch dimension followed by the shape
            of an image in the stack.
        """
        # Set the image parameters
        self.parameters = parameters
        # Set the image stack
        self.images = jnp.asarray(images)
