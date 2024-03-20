"""Cryojax compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import pandas as pd
import mrcfile
import dataclasses
import pathlib
from jaxtyping import Shaped
from typing import final, Callable, Any

import equinox as eqx
import jax.numpy as jnp

from ..io import read_and_validate_starfile
from ..simulator import ImageConfig, EulerAnglePose, CTF
from ..typing import RealImage
from ._particle_stack import AbstractParticleStack
from ._dataset import AbstractDataset


RELION_REQUIRED_OPTICS_KEYS = [
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


class RelionParticleStack(AbstractParticleStack):
    """A particle stack with information imported from
    [RELION](https://relion.readthedocs.io/en/release-5.0/).
    """

    image_stack: Shaped[RealImage, "batch_dim"] = eqx.field(converter=jnp.asarray)
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


def default_relion_make_config(
    shape: tuple[int, int], pixel_size: float, **kwargs: Any
):
    return ImageConfig(shape, pixel_size, **kwargs)


@dataclasses.dataclass(frozen=True)
class RelionDataset(AbstractDataset):
    """A dataset that wraps a Relion particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    path_to_relion_project: pathlib.Path
    data_blocks: dict[str, pd.DataFrame]

    make_config: Callable[[tuple[int, int], float], ImageConfig]

    @final
    def __init__(
        self,
        path_to_starfile: str | pathlib.Path,
        path_to_relion_project: str | pathlib.Path,
        make_config: Callable[
            [tuple[int, int], float], ImageConfig
        ] = default_relion_make_config,
    ):
        """**Arguments:**

        - `path_to_starfile`: The path to the Relion STAR file.
        - `path_to_relion_project`: The path to the Relion project directory.
        """
        data_blocks = read_and_validate_starfile(path_to_starfile)
        _validate_relion_data_blocks(data_blocks)
        object.__setattr__(self, "data_blocks", data_blocks)
        object.__setattr__(
            self, "path_to_relion_project", pathlib.Path(path_to_relion_project)
        )
        object.__setattr__(self, "make_config", make_config)

    @final
    def __getitem__(self, index) -> RelionParticleStack:
        # Load particle data and optics group
        particle_blocks = self.data_blocks["particles"].iloc[index]
        optics_group = self.data_blocks["optics"].iloc[0]
        # Load particle image stack rlnImageName
        relion_particle_index, image_stack_filename = particle_blocks[
            "rlnImageName"
        ].split("@")
        path_to_image_stack = pathlib.Path(
            self.path_to_relion_project, image_stack_filename
        )
        particle_index = int(relion_particle_index) - 1
        with mrcfile.mmap(path_to_image_stack, mode="r", permissive=True) as mrc:
            image_stack = mrc.data[particle_index]  # type: ignore
        # Read metadata into a RelionParticleStack
        # ... particle data
        defocus_u_in_angstroms = particle_blocks["rlnDefocusU"]
        defocus_v_in_angstroms = particle_blocks["rlnDefocusV"]
        astigmatism_angle = particle_blocks["rlnDefocusAngle"]
        phase_shift = particle_blocks["rlnPhaseShift"]
        # ... optics group data
        image_size = optics_group["rlnImageSize"]
        pixel_size = optics_group["rlnImagePixelSize"]
        voltage_in_kilovolts = optics_group["rlnVoltage"]
        spherical_aberration_in_mm = optics_group["rlnVoltage"]
        amplitude_contrast_ratio = optics_group["rlnAmplitudeContrast"]
        # ... create cryojax objects
        config = self.make_config((image_size, image_size), pixel_size)
        ctf = CTF(
            defocus_u_in_angstroms,
            defocus_v_in_angstroms,
            astigmatism_angle,
            voltage_in_kilovolts,
            spherical_aberration_in_mm,
            amplitude_contrast_ratio,
            phase_shift,
        )
        pose = EulerAnglePose()
        # ... fill objects with optional keys
        particle_keys = particle_blocks.keys()
        pose_parameter_names_and_values = []
        if "rlnOriginXAngst" in particle_keys:
            pose_parameter_names_and_values.append(
                ("offset_x_in_angstroms", particle_blocks["rlnOriginXAngst"])
            )
        if "rlnOriginYAngst" in particle_keys:
            pose_parameter_names_and_values.append(
                ("offset_y_in_angstroms", particle_blocks["rlnOriginYAngst"])
            )
        if "rlnAngleRot" in particle_keys:
            pose_parameter_names_and_values.append(
                ("view_phi", particle_blocks["rlnAngleRot"])
            )
        if "rlnAngleTilt" in particle_keys:
            pose_parameter_names_and_values.append(
                ("view_theta", particle_blocks["rlnAngleTilt"])
            )
        if "rlnAnglePsi" in particle_keys:
            pose_parameter_names_and_values.append(
                ("view_psi", particle_blocks["rlnAnglePsi"])
            )
        pose_parameter_names, pose_parameter_values = tuple(
            zip(*pose_parameter_names_and_values)
        )
        pose = eqx.tree_at(
            lambda p: tuple([getattr(p, name) for name in pose_parameter_names]),
            pose,
            tuple(pose_parameter_values),
        )

        return RelionParticleStack(image_stack, config, pose, ctf)

    @final
    def __len__(self) -> int:
        return len(self.data_blocks["particles"])


def _validate_relion_data_blocks(data_blocks: dict[str, pd.DataFrame]):
    if "particles" not in data_blocks.keys():
        raise ValueError("Missing key 'particles' in `starfile.read` output.")
    else:
        if not set(RELION_REQUIRED_PARTICLE_KEYS).issubset(
            set(data_blocks["particles"].keys())
        ):
            raise ValueError(
                "Missing required keys in starfile 'particles' group. "
                f"Required keys are {RELION_REQUIRED_PARTICLE_KEYS}."
            )
    if "optics" not in data_blocks.keys():
        raise ValueError("Missing key 'optics' in `starfile.read` output.")
    else:
        if not set(RELION_REQUIRED_OPTICS_KEYS).issubset(
            set(data_blocks["optics"].keys())
        ):
            raise ValueError(
                "Missing required keys in starfile 'optics' group. "
                f"Required keys are {RELION_REQUIRED_OPTICS_KEYS}."
            )
