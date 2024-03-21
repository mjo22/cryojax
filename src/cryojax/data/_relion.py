"""Cryojax compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import pandas as pd
import mrcfile
import dataclasses
import pathlib
from jaxtyping import Shaped, Float
from typing import final, Callable, Any

import equinox as eqx
import numpy as np
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

    image_stack: Shaped[RealImage, "..."]
    config: ImageConfig
    pose: EulerAnglePose
    ctf: CTF

    def __init__(
        self,
        image_stack: Shaped[RealImage, "..."] | Float[np.ndarray, "... Ny Nx"],
        config: ImageConfig,
        pose: EulerAnglePose,
        ctf: CTF,
    ):
        # Set image stack and config as is
        self.image_stack = jnp.asarray(image_stack)
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
    shape: tuple[int, int], pixel_size: float | Float[np.ndarray, "..."], **kwargs: Any
):
    return ImageConfig(shape, jnp.asarray(pixel_size), **kwargs)


@dataclasses.dataclass(frozen=True)
class RelionDataset(AbstractDataset):
    """A dataset that wraps a Relion particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    path_to_relion_project: pathlib.Path
    data_blocks: dict[str, pd.DataFrame]

    make_config: Callable[
        [tuple[int, int], float | Float[np.ndarray, "..."]], ImageConfig
    ]

    @final
    def __init__(
        self,
        path_to_starfile: str | pathlib.Path,
        path_to_relion_project: str | pathlib.Path,
        make_config: Callable[
            [tuple[int, int], float | Float[np.ndarray, "..."]], ImageConfig
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
    def __getitem__(self, index: int | slice) -> RelionParticleStack:
        # Load particle data and optics group
        particle_blocks = self.data_blocks["particles"].iloc[index]
        optics_group = self.data_blocks["optics"].iloc[0]
        # Load particle image stack rlnImageName
        image_stack_index_and_name_series_or_str = particle_blocks["rlnImageName"]
        if isinstance(image_stack_index_and_name_series_or_str, str):
            # In this block, the user most likely used standard indexing, like
            # `dataset = RelionDataset(...); particle_stack = dataset[1]`
            image_stack_index_and_name_str = image_stack_index_and_name_series_or_str
            # ... split the whole string into its image index and filename
            relion_particle_index, image_stack_filename = (
                image_stack_index_and_name_str.split("@")
            )
            # ... create full path to the image stack
            path_to_image_stack = pathlib.Path(
                self.path_to_relion_project, image_stack_filename
            )
            # ... relion convention starts indexing at 1, not 0
            particle_index = np.asarray(relion_particle_index, dtype=int) - 1
        elif isinstance(image_stack_index_and_name_series_or_str, pd.Series):
            # In this block, the user most likely used fancy indexing, like
            # `dataset = RelionDataset(...); particle_stack = dataset[1:10]`
            image_stack_index_and_name_series = image_stack_index_and_name_series_or_str
            # ... split the pandas.Series into a pandas.DataFrame with two columns: one for
            # the image index and another for the filename
            image_stack_index_and_name_dataframe = (
                image_stack_index_and_name_series.str.split("@", expand=True)
            )
            # ... get a pandas.Series for each the index and the filename
            relion_particle_index, image_stack_filename = [
                image_stack_index_and_name_dataframe[column]
                for column in image_stack_index_and_name_dataframe.columns
            ]
            # ... multiple filenames in the same STAR file is not supported with
            # fancy indexing
            if image_stack_filename.nunique() != 1:
                raise ValueError(
                    "Found multiple image stack filenames when reading "
                    "STAR file rows. This is most likely because you tried to "
                    "use fancy indexing with multiple image stack filenames "
                    "in the same STAR file. If a STAR file refers to multiple image "
                    "stack filenames, fancy indexing is not supported. For example, this will "
                    "raise an error: `dataset = RelionDataset(...); particle_stack = dataset[1:10]`."
                )
            # ... create full path to the image stack
            path_to_image_stack = pathlib.Path(
                self.path_to_relion_project, image_stack_filename[0]
            )
            # ... relion convention starts indexing at 1, not 0
            particle_index = (
                np.asarray(relion_particle_index.astype(int), dtype=int) - 1
            )
        else:
            raise IOError(
                "Could not read `rlnImageName` in STAR file for `RelionDataset` "
                f"index equal to {index}."
            )
        with mrcfile.mmap(path_to_image_stack, mode="r", permissive=True) as mrc:
            image_stack = np.asarray(mrc.data[particle_index])  # type: ignore
        # Read metadata into a RelionParticleStack
        # ... particle data
        defocus_u_in_angstroms = np.asarray(particle_blocks["rlnDefocusU"])
        defocus_v_in_angstroms = np.asarray(particle_blocks["rlnDefocusV"])
        astigmatism_angle = np.asarray(particle_blocks["rlnDefocusAngle"])
        phase_shift = np.asarray(particle_blocks["rlnPhaseShift"])
        # ... optics group data
        image_size = np.asarray(optics_group["rlnImageSize"])
        pixel_size = np.asarray(optics_group["rlnImagePixelSize"])
        voltage_in_kilovolts = np.asarray(optics_group["rlnVoltage"])
        spherical_aberration_in_mm = np.asarray(optics_group["rlnVoltage"])
        amplitude_contrast_ratio = np.asarray(optics_group["rlnAmplitudeContrast"])
        # ... create cryojax objects
        config = self.make_config((int(image_size), int(image_size)), pixel_size)
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
        # ... values for the pose are optional, so look to see if
        # each key is present
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
            # Relion uses -999.0 as a placeholder for an un-estimated in-plane
            # rotation
            if isinstance(particle_blocks["rlnAnglePsi"], pd.Series):
                # ... check if all values are equal to -999.0. If so, just
                # replace the whole pandas.Series with 0.0
                if (
                    particle_blocks["rlnAnglePsi"].nunique() == 1
                    and particle_blocks["rlnAnglePsi"][0] == -999.0
                ):
                    particle_blocks_for_psi = 0.0
                # ... otherwise, replace -999.0 values with 0.0
                else:
                    particle_blocks_for_psi = particle_blocks["rlnAnglePsi"].where(
                        lambda x: x != -999.0, 0.0
                    )
            else:
                # ... if the column is just equal to a float, then
                # directly check if it is equal to -999.0
                particle_blocks_for_psi = (
                    0.0
                    if particle_blocks["rlnAnglePsi"] == -999.0
                    else particle_blocks["rlnAnglePsi"]
                )
            pose_parameter_names_and_values.append(
                ("view_psi", particle_blocks_for_psi)
            )
        pose_parameter_names, pose_parameter_values = tuple(
            zip(*pose_parameter_names_and_values)
        )
        # ... fill the EulerAnglePose will keys that are present. if they are not
        # present, keep the default values in the `pose = EulerAnglePose()` instantiation
        pose = eqx.tree_at(
            lambda p: tuple([getattr(p, name) for name in pose_parameter_names]),
            pose,
            tuple([jnp.asarray(value) for value in pose_parameter_values]),
        )

        return RelionParticleStack(image_stack, config, pose, ctf)

    @final
    def __len__(self) -> int:
        return len(self.data_blocks["particles"])


@dataclasses.dataclass(frozen=True)
class HelicalRelionDataset(AbstractDataset):
    """A wrapped `RelionDataset` to read helical parameters."""

    dataset: RelionDataset

    @final
    def __init__(
        self,
        dataset: RelionDataset,
    ):
        """**Arguments:**

        - `dataset`: The wrappped `RelionDataset`. This will be slightly
                     modified to read helical parameters.
        """
        object.__setattr__(self, "dataset", dataset)

    @final
    def __getitem__(
        self, index: int | slice | tuple[int, int] | tuple[slice, slice]
    ) -> RelionParticleStack:
        dataset = self.dataset[index]
        return dataset

    @final
    def __len__(self) -> int:
        return len(self.dataset)


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
