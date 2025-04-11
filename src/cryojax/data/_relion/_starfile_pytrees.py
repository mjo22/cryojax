from typing import Optional

from ...simulator import ContrastTransferTheory, EulerAnglePose, InstrumentConfig
from .._particle_data import AbstractParticleParameters


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
