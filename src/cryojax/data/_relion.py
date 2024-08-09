"""Cryojax compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import dataclasses
import pathlib
from typing import Any, Callable, final, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Int

from ..io import read_and_validate_starfile
from ..simulator import ContrastTransferFunction, EulerAnglePose, InstrumentConfig
from ._dataset import AbstractDataset
from ._particle_stack import AbstractParticleStack


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

    instrument_config: InstrumentConfig
    pose: EulerAnglePose
    ctf: ContrastTransferFunction
    image_stack: Optional[Float[Array, "... y_dim x_dim"]]

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        pose: EulerAnglePose,
        ctf: ContrastTransferFunction,
        image_stack: Optional[Float[Array, "... y_dim x_dim"]] = None,
    ):
        """**Arguments:**

        - `instrument_config`:
            The instrument configuration. Any subset of pytree leaves may
            have a batch dimension.
        - `pose`:
            The pose, represented by euler angles. Any subset of pytree leaves may
            have a batch dimension. Upon instantiation, `pose.offset_z_in_angstroms`
            is set to zero.
        - `ctf`:
            The contrast transfer function. Any subset of pytree leaves may
            have a batch dimension. Upon instantiation,
            `ctf.defocus_in_angstroms` is set to
            `ctf.defocus_in_angstroms + pose.offset_z_in_angstroms`.
        - `image_stack`:
            The stack of images. The shape of this array
            is a leading batch dimension followed by the shape
            of an image in the stack.
        """
        # Set instrument config as is
        self.instrument_config = instrument_config
        # Set CTF using the defocus offset in the EulerAnglePose
        self.ctf = eqx.tree_at(
            lambda tf: tf.defocus_in_angstroms,
            ctf,
            ctf.defocus_in_angstroms + pose.offset_z_in_angstroms,
        )
        # Set defocus offset to zero
        self.pose = eqx.tree_at(lambda pose: pose.offset_z_in_angstroms, pose, 0.0)
        # Optionally set image stack
        self.image_stack = None if image_stack is None else jnp.asarray(image_stack)


def _default_make_instrument_config_fn(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    voltage_in_kilovolts: Float[Array, ""],
    **kwargs: Any,
):
    return InstrumentConfig(shape, pixel_size, voltage_in_kilovolts, **kwargs)


@dataclasses.dataclass(frozen=True)
class RelionDataset(AbstractDataset):
    """A dataset that wraps a Relion particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    path_to_relion_project: pathlib.Path
    data_blocks: dict[str, pd.DataFrame]
    make_instrument_config_fn: Callable[
        [tuple[int, int], Float[Array, "..."], Float[Array, "..."]], InstrumentConfig
    ]
    get_image_stack: bool
    get_cpu_arrays: bool

    @final
    def __init__(
        self,
        path_to_starfile: str | pathlib.Path,
        path_to_relion_project: str | pathlib.Path,
        get_image_stack: bool = True,
        get_cpu_arrays: bool = False,
        make_instrument_config_fn: Callable[
            [tuple[int, int], Float[Array, "..."], Float[Array, "..."]],
            InstrumentConfig,
        ] = _default_make_instrument_config_fn,
    ):
        """**Arguments:**

        - `path_to_starfile`: The path to the Relion STAR file.
        - `path_to_relion_project`: The path to the Relion project directory.
        - `get_image_stack`:
            If `True`, read the stack of images from the STAR file. Otherwise,
            just read parameters.
        - `get_cpu_arrays`:
            If `True`, force that JAX arrays are loaded on the CPU. If `False`,
            load on the default device.
        - `make_instrument_config_fn`:
            A function used for `InstrumentConfig` initialization that returns
            an `InstrumentConfig`. This is used to customize the metadata of the
            read object.
        """
        data_blocks = read_and_validate_starfile(path_to_starfile)
        _validate_relion_data_blocks(data_blocks)
        object.__setattr__(self, "data_blocks", data_blocks)
        object.__setattr__(
            self, "path_to_relion_project", pathlib.Path(path_to_relion_project)
        )
        object.__setattr__(self, "make_instrument_config_fn", make_instrument_config_fn)
        object.__setattr__(self, "get_image_stack", get_image_stack)
        object.__setattr__(self, "get_cpu_arrays", get_cpu_arrays)

    @final
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"]
    ) -> RelionParticleStack:
        # Load particle data and optics group
        n_rows = self.data_blocks["particles"].shape[0]
        index_error_msg = lambda idx: (
            "The index at which the `RelionDataset` was accessed was out of bounds! "
            f"The number of rows in the dataset is {n_rows}, but you tried to "
            f"access the index {idx}."
        )
        # ... pandas has bad error messages for its indexing
        if isinstance(index, (int, Int[np.ndarray, ""])):  # type: ignore
            if index > n_rows - 1:
                raise IndexError(index_error_msg(index))
        elif isinstance(index, slice):
            if index.start is not None and index.start > n_rows - 1:
                raise IndexError(index_error_msg(index.start))
        elif isinstance(index, np.ndarray):
            pass  # ... catch exceptions later
        else:
            raise IndexError(
                f"Indexing with the type {type(index)} is not supported by "
                "`RelionDataset`. Indexing by integers is supported, one-dimensional "
                "fancy indexing is supported, and numpy-array indexing is supported. "
                "For example, like `particle = dataset[0]`, "
                "`particle_stack = dataset[0:5]`, "
                "or `particle_stack = dataset[np.array([1, 4, 3, 2])]`."
            )
        try:
            particle_blocks = self.data_blocks["particles"].iloc[index]
        except Exception:
            raise IndexError(
                "Error when indexing the `pandas.Dataframe` for the particle stack "
                "from the `starfile.read` output."
            )
        optics_group = self.data_blocks["optics"].iloc[0]
        # Load the image stack and STAR file parameters. First, get the device
        # on which to load arrays
        device = jax.devices("cpu")[0] if self.get_cpu_arrays else None
        # ... load stack of images, unless otherwise specified
        image_stack = (
            self._get_image_stack(index, particle_blocks, device)
            if self.get_image_stack
            else None
        )
        # ... load image parameters into cryoJAX objects
        instrument_config, ctf, pose = self._get_starfile_params(
            particle_blocks,
            optics_group,
            device,
        )

        return RelionParticleStack(instrument_config, pose, ctf, image_stack)

    @final
    def __len__(self) -> int:
        return len(self.data_blocks["particles"])

    def _get_starfile_params(
        self, particle_blocks, optics_group, device
    ) -> tuple[InstrumentConfig, ContrastTransferFunction, EulerAnglePose]:
        defocus_in_angstroms = jnp.asarray(particle_blocks["rlnDefocusU"], device=device)
        astigmatism_in_angstroms = jnp.asarray(
            particle_blocks["rlnDefocusV"], device=device
        ) - jnp.asarray(particle_blocks["rlnDefocusU"], device=device)
        astigmatism_angle = jnp.asarray(particle_blocks["rlnDefocusAngle"], device=device)
        phase_shift = jnp.asarray(particle_blocks["rlnPhaseShift"])
        # ... optics group data
        image_size = jnp.asarray(optics_group["rlnImageSize"], device=device)
        pixel_size = jnp.asarray(optics_group["rlnImagePixelSize"], device=device)
        voltage_in_kilovolts = float(optics_group["rlnVoltage"])  # type: ignore
        spherical_aberration_in_mm = jnp.asarray(
            optics_group["rlnSphericalAberration"], device=device
        )
        amplitude_contrast_ratio = jnp.asarray(
            optics_group["rlnAmplitudeContrast"], device=device
        )
        # ... create cryojax objects
        instrument_config = self.make_instrument_config_fn(
            (int(image_size), int(image_size)),
            pixel_size,
            jnp.asarray(voltage_in_kilovolts, device=device),
        )
        ctf = ContrastTransferFunction(
            defocus_in_angstroms=defocus_in_angstroms,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
            astigmatism_angle=astigmatism_angle,
            voltage_in_kilovolts=voltage_in_kilovolts,
            spherical_aberration_in_mm=spherical_aberration_in_mm,
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            phase_shift=phase_shift,
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
        elif "rlnAngleTiltPrior" in particle_keys:  # support for helices
            pose_parameter_names_and_values.append(
                ("view_theta", particle_blocks["rlnAngleTiltPrior"])
            )
        if "rlnAnglePsi" in particle_keys:
            # Relion uses -999.0 as a placeholder for an un-estimated in-plane
            # rotation
            if isinstance(particle_blocks["rlnAnglePsi"], pd.Series):
                # ... check if all values are equal to -999.0. If so, just
                # replace the whole pandas.Series with 0.0
                if (
                    particle_blocks["rlnAnglePsi"].nunique() == 1
                    and particle_blocks["rlnAnglePsi"].iloc[0] == -999.0
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
            pose_parameter_names_and_values.append(("view_psi", particle_blocks_for_psi))
        elif "rlnAnglePsiPrior" in particle_keys:  # support for helices
            pose_parameter_names_and_values.append(
                ("view_psi", particle_blocks["rlnAnglePsiPrior"])
            )
        pose_parameter_names, pose_parameter_values = tuple(
            zip(*pose_parameter_names_and_values)
        )
        # ... fill the EulerAnglePose will keys that are present. if they are not
        # present, keep the default values in the `pose = EulerAnglePose()`
        # instantiation
        pose = eqx.tree_at(
            lambda p: tuple([getattr(p, name) for name in pose_parameter_names]),
            pose,
            tuple([jnp.asarray(value, device=device) for value in pose_parameter_values]),
        )

        return instrument_config, ctf, pose

    def _get_image_stack(
        self,
        index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"],
        particle_blocks,
        device,
    ) -> Float[Array, "... y_dim x_dim"]:
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
            # ... split the pandas.Series into a pandas.DataFrame with two columns:
            # one for the image index and another for the filename
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
                    "stack filenames, fancy indexing is not supported. For example, "
                    "this will raise an error: `dataset = RelionDataset(...); "
                    "particle_stack = dataset[1:10]`."
                )
            # ... create full path to the image stack
            path_to_image_stack = pathlib.Path(
                self.path_to_relion_project,
                np.asarray(image_stack_filename, dtype=object)[0],
            )
            # ... relion convention starts indexing at 1, not 0
            particle_index = np.asarray(relion_particle_index.astype(int), dtype=int) - 1
        else:
            raise IOError(
                "Could not read `rlnImageName` in STAR file for `RelionDataset` "
                f"index equal to {index}."
            )

        with mrcfile.mmap(path_to_image_stack, mode="r", permissive=True) as mrc:
            image_stack = np.asarray(mrc.data[particle_index])  # type: ignore

        return jnp.asarray(image_stack, device=device)


@dataclasses.dataclass(frozen=True)
class HelicalRelionDataset(AbstractDataset):
    """A wrapped `RelionDataset` to read helical tubes.

    In particular, a `HelicalRelionDataset` indexes one
    helical filament at a time. For example, after manual
    particle picking in RELION, we can index a particular filament
    with

    ```python
    # Read in a STAR file particle stack
    dataset = RelionDataset(...)
    helical_dataset = HelicalRelionDataset(dataset)
    # ... get a particle stack for a filament
    particle_stack_for_a_filament = helical_dataset[0]
    # ... get a particle stack for another filament
    particle_stack_for_another_filament = helical_dataset[1]
    ```

    Unlike a `RelionDataset`, a `HelicalRelionDataset` does not
    support fancy indexing.
    """

    dataset: RelionDataset
    n_filaments: int
    n_filaments_per_micrograph: Int[np.ndarray, " n_micrographs"]
    micrograph_names: list[str]

    @final
    def __init__(
        self,
        dataset: RelionDataset,
    ):
        """**Arguments:**

        - `dataset`: The wrappped `RelionDataset`. This will be slightly
                     modified to read one helix at a time, rather than one
                     image crop at a time.
        """
        # Validate the STAR file and store the RelionDataset
        _validate_helical_relion_data_blocks(dataset.data_blocks)
        object.__setattr__(self, "dataset", dataset)
        # Compute and store the number of filaments, number of filaments per micrograph
        # and micrograph names
        n_filaments_per_micrograph, micrograph_names = (
            _get_number_of_filaments_per_micrograph_in_helical_data_blocks(
                dataset.data_blocks
            )
        )
        object.__setattr__(self, "n_filaments", int(np.sum(n_filaments_per_micrograph)))
        object.__setattr__(self, "n_filaments_per_micrograph", n_filaments_per_micrograph)
        object.__setattr__(self, "micrograph_names", micrograph_names)

    def get_data_blocks_at_filament_index(
        self, filament_index: int | Int[np.ndarray, ""]
    ) -> pd.DataFrame:
        # Map the filament index to a micrograph index
        last_index_of_filament_per_micrograph = (
            np.cumsum(self.n_filaments_per_micrograph) - 1
        )
        micrograph_index = np.where(
            last_index_of_filament_per_micrograph >= filament_index
        )[0].min()
        # Get the filament index in this particular micrograph
        filament_index_in_micrograph = (
            self.n_filaments_per_micrograph[micrograph_index] - 1
        ) - (last_index_of_filament_per_micrograph[micrograph_index] - filament_index)
        # .. get the data blocks only at the filament corresponding to the filament index
        particle_data_blocks = self.dataset.data_blocks["particles"]
        particle_data_blocks_at_micrograph = particle_data_blocks[
            particle_data_blocks["rlnMicrographName"]
            == self.micrograph_names[micrograph_index]
        ]
        particle_data_blocks_at_filament = particle_data_blocks_at_micrograph[
            particle_data_blocks_at_micrograph["rlnHelicalTubeID"]
            == filament_index_in_micrograph + 1
        ]

        return particle_data_blocks_at_filament

    @final
    def __getitem__(
        self, filament_index: int | Int[np.ndarray, ""]
    ) -> RelionParticleStack:
        if not isinstance(filament_index, (int, Int[np.ndarray, ""])):  # type: ignore
            raise IndexError(
                "When indexing a `HelicalRelionDataset`, only "
                f"python or numpy-like integer indices are supported, such as "
                "`helical_particle_stack = helical_dataset[3]`. "
                f"Got index {filament_index} of type {type(filament_index)}."
            )
        # Make sure the filament index is in-bounds
        if filament_index + 1 > self.n_filaments:
            raise IndexError(
                "The index at which the `HelicalRelionDataset` was accessed was out of "
                f"bounds! The number of filaments in the dataset is {self.n_filaments}, "
                f"but you tried to access the index {filament_index}."
            )
        # Get the particle stack indices corresponding to this filament
        particle_data_blocks_at_filament = self.get_data_blocks_at_filament_index(
            filament_index
        )
        particle_indices = np.asarray(particle_data_blocks_at_filament.index, dtype=int)
        # Access the particle stack at these indices
        dataset = self.dataset[particle_indices]
        return dataset

    @final
    def __len__(self) -> int:
        return self.n_filaments


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


def _validate_helical_relion_data_blocks(data_blocks: dict[str, pd.DataFrame]):
    particle_data_blocks = data_blocks["particles"]
    if "rlnHelicalTubeID" not in particle_data_blocks.columns:
        raise ValueError(
            "Missing column 'rlnHelicalTubeID' in `starfile.read` output. "
            "This column must be present when using a `HelicalRelionDataset`."
        )


def _get_number_of_filaments_per_micrograph_in_helical_data_blocks(
    data_blocks: dict[str, pd.DataFrame],
) -> tuple[Int[np.ndarray, " n_micrographs"], list[str]]:
    particle_data_blocks = data_blocks["particles"]
    micrograph_names = particle_data_blocks["rlnMicrographName"].unique().tolist()
    n_filaments_per_micrograph = np.asarray(
        tuple(
            particle_data_blocks[
                particle_data_blocks["rlnMicrographName"] == micrograph_name
            ]["rlnHelicalTubeID"].max()
            for micrograph_name in micrograph_names
        ),
        dtype=int,
    )

    return n_filaments_per_micrograph, micrograph_names
