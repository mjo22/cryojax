"""Cryojax compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import pandas as pd
import mrcfile
import dataclasses
from jaxtyping import Shaped
from os import PathLike

from ..io import read_and_validate_starfile
from ..simulator import ImageConfig, EulerAnglePose, CTF
from ..typing import RealImage
from ._particle_stack import AbstractParticleStack
from ._dataset import AbstractDataset


class RelionParticleStack(AbstractParticleStack):
    """A particle stack with information imported from
    [RELION](https://relion.readthedocs.io/en/release-5.0/).
    """

    image_stack: Shaped[RealImage, "batch_dim"]
    config: ImageConfig
    pose: EulerAnglePose
    ctf: CTF


RelionParticleStack.__init__.__doc__ = """**Arguments:**

- `image_stack`: The stack of images. The shape of this array
                 is a leading batch dimension followed by the shape
                 of an image in the stack.
- `config`: The image configuration. Any subset of pytree leaves may
            have a batch dimension.
- `pose`: The pose, represented by euler angles. Any subset of pytree leaves may
          have a batch dimension.
- `ctf`: The contrast transfer function. Any subset of pytree leaves may
         have a batch dimension.
"""


@dataclasses.dataclass(frozen=True)
class RelionDataset(AbstractDataset):
    """A dataset that wraps a Relion particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    starfile_dataframe: pd.DataFrame

    def __init__(self, path_to_starfile: PathLike):
        """**Arguments:**

        - `path_to_starfile`: The path to the Relion STAR file.
        """
        object.__setattr__(
            self, "starfile_dataframe", read_and_validate_starfile(path_to_starfile)
        )

    def __getitem__(self, index) -> RelionParticleStack:
        return RelionParticleStack(None, None, None, None)

    def __len__(self) -> int:
        return 0
