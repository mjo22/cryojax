"""Cryojax compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import pandas as pd
import mrcfile
import dataclasses
import pathlib
from jaxtyping import Shaped
from typing import final

import equinox as eqx
import jax.numpy as jnp

from ..io import read_and_validate_starfile
from ..simulator import ImageConfig, EulerAnglePose, CTF
from ..typing import RealImage
from ._particle_stack import AbstractParticleStack
from ._dataset import AbstractDataset


RELION_OPTICS_KEYS = [
    "rlnImageSize",
    "rlnVoltage",
    "rlnImagePixelSize",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
]
RELION_REQUIRED_PARTICLE_KEYS = [
    "rlnDefocusU",
    "rlnDefocusV",
    "rlnDefocusAngle",
    "rlnPhaseShift",
    "rlnImageName",
]
RELION_OPTIONAL_PARTICLE_KEYS = [
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
]


class RelionParticleStack(AbstractParticleStack):
    """A particle stack with information imported from
    [RELION](https://relion.readthedocs.io/en/release-5.0/).
    """

    image_stack: Shaped[RealImage, "batch_dim"]
    config: ImageConfig
    pose: EulerAnglePose
    ctf: CTF

    def __init__(
        self,
        image_stack: Shaped[RealImage, "batch_dim"],
        config: ImageConfig,
        pose: EulerAnglePose,
        ctf: CTF,
    ):
        # Set image stack and config as is
        self.image_stack = image_stack
        self.config = config
        # Set CTF using the defocus offset in the EulerAnglePose
        self.ctf = eqx.tree_at(
            lambda ctf: (ctf.defocus_u_in_angstroms, ctf.defocus_v_in_angstroms),
            ctf,
            (
                ctf.defocus_u_in_angstroms + pose.offset_z_in_angstroms,
                ctf.defocus_v_in_angstroms + pose.offset_z_in_angstroms,
            ),
        )
        # Set defocus offset to zero
        self.pose = eqx.tree_at(
            lambda pose: pose.offset_z_in_angstroms, pose, jnp.asarray(0.0)
        )


RelionParticleStack.__init__.__doc__ = """**Arguments:**

- `image_stack`: The stack of images. The shape of this array
                 is a leading batch dimension followed by the shape
                 of an image in the stack.
- `config`: The image configuration. Any subset of pytree leaves may
            have a batch dimension.
- `pose`: The pose, represented by euler angles. Any subset of pytree leaves may
          have a batch dimension. Upon instantiation, `pose.offset_z_in_angstroms`
          is set to zero.
- `ctf`: The contrast transfer function. Any subset of pytree leaves may
         have a batch dimension. Upon instantiation, `ctf.defocus_u_in_angstroms`
         is set to `ctf.defocus_u_in_angstroms + pose.offset_z_in_angstroms` (and
         also for `ctf.defocus_v_in_angstroms`).
"""


@dataclasses.dataclass(frozen=True)
class RelionDataset(AbstractDataset):
    """A dataset that wraps a Relion particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    data_blocks: dict[str, pd.DataFrame]

    @final
    def __init__(self, path_to_starfile: str | pathlib.Path):
        """**Arguments:**

        - `path_to_starfile`: The path to the Relion STAR file.
        """
        data_blocks = read_and_validate_starfile(path_to_starfile)
        _validate_relion_data_blocks(data_blocks)
        object.__setattr__(self, "data_blocks", data_blocks)

    @final
    def __getitem__(self, index) -> RelionParticleStack:
        return RelionParticleStack(None, None, None, None)

    @final
    def __len__(self) -> int:
        return len(self.data_blocks["particles"])


def _validate_relion_data_blocks(data_blocks: dict[str, pd.DataFrame]):
    if "particles" not in data_blocks.keys():
        raise Exception("Missing particles in starfile")
    if "optics" not in data_blocks.keys():
        raise Exception("Missing optics in starfile")
