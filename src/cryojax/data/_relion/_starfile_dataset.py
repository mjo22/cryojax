"""cryoJAX compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import abc
import pathlib
import re
import warnings
from typing import Any, Callable, Literal, Optional, TypedDict
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
import starfile
from jaxtyping import Array, Float, Int

from ...image.operators import Constant, FourierGaussian
from ...internal import NDArrayLike
from ...io import read_and_validate_starfile, write_image_stack_to_mrc
from ...simulator import (
    AberratedAstigmaticCTF,
    ContrastTransferTheory,
    EulerAnglePose,
    InstrumentConfig,
)
from .._particle_data import (
    AbstractParticleParameterDataset,
    AbstractParticleStackDataset,
)
from ._starfile_pytrees import RelionParticleParameters, RelionParticleStack


RELION_REQUIRED_OPTICS_KEYS = [
    "rlnImageSize",
    "rlnVoltage",
    "rlnImagePixelSize",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnOpticsGroup",
]
RELION_REQUIRED_PARTICLE_KEYS = [
    "rlnDefocusU",
    "rlnDefocusV",
    "rlnDefocusAngle",
    "rlnPhaseShift",
    # "rlnImageName",
    "rlnOpticsGroup",
]
RELION_POSE_PARTICLE_KEYS = [
    "rlnOriginXAngst",
    "rlnOriginYAngst",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
]


class StarfileData(TypedDict):
    particles: pd.DataFrame
    optics: pd.DataFrame


class ImageFilenameSettings(TypedDict):
    prefix: str
    output_folder: str | pathlib.Path
    n_characters: int
    delimiter: str


def _default_make_config_fn(
    shape: tuple[int, int],
    pixel_size: Float[Array, ""],
    voltage_in_kilovolts: Float[Array, ""],
    **kwargs: Any,
):
    return InstrumentConfig(shape, pixel_size, voltage_in_kilovolts, **kwargs)


class AbstractRelionParticleParameterDataset(
    AbstractParticleParameterDataset[RelionParticleParameters]
):
    @property
    @abc.abstractmethod
    def path_to_starfile(self) -> pathlib.Path:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def starfile_data(self) -> StarfileData:
        raise NotImplementedError

    @starfile_data.setter
    @abc.abstractmethod
    def starfile_data(self, value: dict[str, pd.DataFrame]):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loads_metadata(self) -> bool:
        raise NotImplementedError

    @loads_metadata.setter
    @abc.abstractmethod
    def loads_metadata(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loads_envelope(self) -> bool:
        raise NotImplementedError

    @loads_envelope.setter
    @abc.abstractmethod
    def loads_envelope(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def broadcasts_optics_group(self) -> bool:
        raise NotImplementedError

    @broadcasts_optics_group.setter
    @abc.abstractmethod
    def broadcasts_optics_group(self, value: bool):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def updates_optics_group(self) -> bool:
        raise NotImplementedError

    @updates_optics_group.setter
    @abc.abstractmethod
    def updates_optics_group(self, value: bool):
        raise NotImplementedError


class RelionParticleParameterDataset(AbstractRelionParticleParameterDataset):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html)
    format.
    """

    def __init__(
        self,
        path_to_starfile: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        overwrite: bool = False,
        *,
        loads_metadata: bool = False,
        broadcasts_optics_group: bool = True,
        loads_envelope: bool = False,
        updates_optics_group: bool = False,
        make_config_fn: Callable[
            [tuple[int, int], Float[Array, "..."], Float[Array, "..."]],
            InstrumentConfig,
        ] = _default_make_config_fn,
    ):
        """**Arguments:**

        - `path_to_starfile`:
            The path to the RELION STAR file. If the path does not exist
            and `mode = 'w'`, an empty dataset will be created.
        - `path_to_relion_project`: The path to the RELION project directory.
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *parameters*. This is done by storing an empty dataset in
            `RelionParticleParameterDataset.starfile_data`. If a STAR file
            already exists at `path_to_starfile`, set `overwrite = True`.
            - If `mode = 'r'`, the STAR file at `path_to_starfile` is read
            into `RelionParticleParameterDataset.starfile_data`.
        - `overwrite`:
            Stores an empty `RelionParticleParameterDataset.starfile_data`
            if `mode = 'w'`.
        - `loads_metadata`:
            If `True`, the resulting `RelionParticleParameters` object loads
            the raw metadata from the STAR file.
            If this is set to `True`, extra care must be taken to make sure that
            `RelionParticleParameters` objects can pass through JIT boundaries
            without recompilation.
        - `broadcasts_optics_group`:
            If `True`, select optics group parameters are broadcasted. If
            there are multiple optics groups in the STAR file, parameters
            are always broadcasted and this option is null.
        - `loads_envelope`:
            If `True`, read in the parameters of the CTF envelope function, i.e.
            "rlnCtfScalefactor" and "rlnCtfBfactor".
        - `updates_optics_group`:
            If `True`, when re-writing STAR file entries via
            `dataset[idx] = parameters` syntax, creates a new optics group entry.
        - `make_config_fn`:
            A function used for `InstrumentConfig` initialization that returns
            an `InstrumentConfig`. This is used to customize the metadata of the
            read object.
        """
        # Private attributes
        self._make_config_fn = make_config_fn
        self._mode = mode
        # Properties without setters
        # ... read starfile and load path
        self._path_to_starfile = pathlib.Path(path_to_starfile)
        self._starfile_data = _load_starfile_data(self._path_to_starfile, mode, overwrite)
        # Properties for loading
        self._loads_metadata = loads_metadata
        self._broadcasts_optics_group = broadcasts_optics_group
        self._loads_envelope = loads_envelope
        # Properties for writing
        self._updates_optics_group = updates_optics_group

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"]
    ) -> RelionParticleParameters:
        # Validate index
        n_rows = self.starfile_data["particles"].shape[0]
        _validate_dataset_index(type(self), index, n_rows)
        # ... read particle data at the requested indices
        particle_data = self.starfile_data["particles"]
        particle_data_at_index = particle_data.iloc[index]
        # ... read optics group given the particle data
        optics_data = self.starfile_data["optics"]
        optics_group = _get_optics_group_from_particle_data(
            particle_data_at_index, optics_data
        )
        # Load the image stack and STAR file parameters
        instrument_config, transfer_theory, pose = _make_pytrees_from_starfile(
            particle_data_at_index,
            optics_group,
            self.broadcasts_optics_group,
            self.loads_envelope,
            self._make_config_fn,
        )
        # ... convert to dataframe for serialization
        if isinstance(particle_data_at_index, pd.Series):
            particle_data_at_index = pd.DataFrame(
                data=particle_data_at_index.values[None, :],
                columns=particle_data_at_index.index,
                index=[0],
            )
        metadata = particle_data_at_index.to_dict() if self.loads_metadata else None
        return RelionParticleParameters(
            instrument_config,
            pose,
            transfer_theory,
            metadata=metadata,
        )

    @override
    def __len__(self) -> int:
        return len(self.starfile_data["particles"])

    @override
    def __setitem__(
        self,
        index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " _"],
        value: RelionParticleParameters,
    ):
        if self.updates_optics_group:
            optics_group_index = _make_optics_group_index(self.starfile_data["optics"])
            particle_data_for_update = _params_to_particle_data(value, optics_group_index)
            optics_data_to_append = _params_to_optics_data(value, optics_group_index)
            optics_data = pd.concat(
                [self.starfile_data["optics"], optics_data_to_append], ignore_index=True
            )
        else:
            particle_data_for_update = _params_to_particle_data(value)
            optics_data = self.starfile_data["optics"]
        particle_data = self.starfile_data["particles"]
        if isinstance(index, (int, np.ndarray)):
            index = np.atleast_1d(index)

        particle_data.loc[
            particle_data.index[index], particle_data_for_update.columns
        ] = particle_data_for_update.values
        self._starfile_data = StarfileData(particles=particle_data, optics=optics_data)

    @override
    def append(self, value: RelionParticleParameters):
        """Add an entry or entries to the dataset.

        **Arguments:**

        - `value`:
            The `RelionParticleParameters` to add to the dataset.
        """
        optics_group_index = _make_optics_group_index(self.starfile_data["optics"])
        optics_data, optics_data_to_append = (
            self.starfile_data["optics"],
            _params_to_optics_data(value, optics_group_index),
        )
        particle_data, particle_data_to_append = (
            self.starfile_data["particles"],
            _params_to_particle_data(value, optics_group_index),
        )
        optics_data = (
            pd.concat([optics_data, optics_data_to_append], ignore_index=True)
            if len(optics_data) > 0
            else optics_data_to_append
        )
        particle_data = (
            pd.concat([particle_data, particle_data_to_append], ignore_index=True)
            if len(particle_data) > 0
            else particle_data_to_append
        )
        self._starfile_data = StarfileData(particles=particle_data, optics=optics_data)

    @override
    def save(
        self,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        path_to_starfile = self.path_to_starfile
        path_exists = path_to_starfile.exists()
        if path_exists and not overwrite:
            raise FileExistsError(
                f"Tried saving STAR file, but file {str(path_to_starfile)} "
                "already exists. To overwrite existing STAR file, set "
                f"`{type(self).__name__}.overwrite = True`."
            )
        else:
            if not path_to_starfile.parent.exists():
                path_to_starfile.parent.mkdir(parents=True)
            starfile.write(self.starfile_data, path_to_starfile, **kwargs)  # type: ignore

    @property
    @override
    def starfile_data(self) -> StarfileData:
        return self._starfile_data

    @starfile_data.setter
    @override
    def starfile_data(self, value: dict[str, pd.DataFrame]):
        if "particles" in value and "optics" in value:
            particle_data, optics_data = value["particles"], value["optics"]
            if isinstance(particle_data, pd.DataFrame) and isinstance(
                optics_data, pd.DataFrame
            ):
                self._starfile_data = StarfileData(
                    particles=particle_data, optics=optics_data
                )
            else:
                raise ValueError(
                    "STAR file data must be a dictionary "
                    "of pandas DataFrames, with keys equal to "
                    "'particles' and 'optics'. Found that the "
                    f"particle data was type `{type(particle_data).__name__}`"
                    "and the optics data was type "
                    f"`{type(optics_data).__name__}`."
                )
        else:
            raise ValueError(
                "STAR file data must be a dictionary "
                "of pandas DataFrames, with keys equal to "
                "'particles' and 'optics'. Tried setting "
                f"with a dictionary with keys {list(value.keys())}."
            )

    @property
    def path_to_starfile(self) -> pathlib.Path:
        return self._path_to_starfile

    @path_to_starfile.setter
    def path_to_starfile(self, value: str | pathlib.Path):
        self._path_to_starfile = pathlib.Path(value)

    @property
    @override
    def loads_metadata(self) -> bool:
        return self._loads_metadata

    @loads_metadata.setter
    @override
    def loads_metadata(self, value: bool):
        self._loads_metadata = value

    @property
    @override
    def loads_envelope(self) -> bool:
        return self._loads_envelope

    @loads_envelope.setter
    @override
    def loads_envelope(self, value: bool):
        self._loads_envelope = value

    @property
    @override
    def broadcasts_optics_group(self) -> bool:
        return self._broadcasts_optics_group

    @broadcasts_optics_group.setter
    @override
    def broadcasts_optics_group(self, value: bool):
        self._broadcasts_optics_group = value

    @property
    @override
    def updates_optics_group(self) -> bool:
        return self._updates_optics_group

    @updates_optics_group.setter
    @override
    def updates_optics_group(self, value: bool):
        self._updates_optics_group = value


class RelionParticleStackDataset(AbstractParticleStackDataset[RelionParticleStack]):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    def __init__(
        self,
        parameter_dataset: AbstractRelionParticleParameterDataset,
        path_to_relion_project: str | pathlib.Path,
        mode: Literal["r", "w"] = "r",
        overwrite: bool = False,
        filename_settings: dict[str, Any] = {},
    ):
        """**Arguments:**

        - `path_to_relion_project`:
            In RELION STAR files, only a relative path is added to the
            'rlnImageName' column. This is relative to the path to the
            "project", which is given by this parameter.
        - `parameter_dataset`:
            The `RelionParticleParameterDataset`.
        - `mode`:
            - If `mode = 'w'`, the dataset is prepared to write new
            *images*. This is done by removing 'rlnImageName' from
            `parameter_dataset.starfile_data`, if it exists at all.
            does not have a column 'rlnImageName' and image files
            are not yet written.
            - If `mode = 'r'`, images are read from the 'rlnImageName'
            stored in the `parameter_dataset.starfile_data`.
        - `overwrite`:
            If `True` and `mode = 'w'`, removes the 'rlnImageName' column
            from `parameter_dataset.starfile_data`.
        - `filename_settings`:
            A dictionary with the following keys:
            - 'prefix':
                A `str` which acts as the prefix to the filenames. If this
                is equal to `"f"`, then the filename for image stack 0 will
                be called "f-00000.mrcs", for `delimiter = '-'` and
                `n_characters = 5`.
                are of format "filenam"
            - 'output_folder':
                A `str` or `pathlib.Path` type where to write MRC files,
                relative to the `path_to_relion_project`.
            - 'n_characters':
                An `int` for the number of characters to write the filename
                number string. If this is equal to `5`, then the filename
                for image stack 0 will be called "f-00000.mrcs", for
                `delimiter = '-'` and `prefix = 'f'`.
             - 'delimiter': str
                A `str` for the delimiter between the filename prefix
                and number string. If this is equal to `'-'`, then the
                filename for image stack 0 will be called "f-00000.mrcs",
                for `n_characters = 5` and `prefix = 'f'`.
        """
        # Set properties for reading image files
        self._path_to_relion_project = pathlib.Path(path_to_relion_project)
        # Set properties for writing image files
        self._filename_settings = _dict_to_filename_settings(filename_settings)
        # For `mode = 'w'`, generate empty 'rlnImageName' column
        if mode == "w":
            particle_data, optics_data = (
                parameter_dataset.starfile_data["particles"],
                parameter_dataset.starfile_data["optics"],
            )
            if "rlnImageName" in particle_data.columns and not overwrite:
                raise IOError(
                    "In `mode = 'w'`, the `RelionParticleStackDataset` "
                    "writes new values for image filenames, i.e. the 'rlnImageName'. "
                    "Found that the STAR file already has a 'rlnImageName' column. "
                    "If you wish to overwrite existing filenames in the STAR file, set "
                    "`overwrite=True`."
                )
            else:
                _set_empty_image_names(particle_data)
                parameter_dataset.starfile_data = dict(
                    optics=optics_data, particles=particle_data
                )
        # The parameter dataset
        self._parameter_dataset = parameter_dataset

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"]
    ) -> RelionParticleStack:
        # ... make sure particle metadata is being loaded
        loads_metadata = self.parameter_dataset.loads_metadata
        self.parameter_dataset.loads_metadata = True
        # ... read parameters
        parameters = self.parameter_dataset[index]
        # ... and construct dataframe
        metadata = parameters.metadata
        particle_dataframe_at_index = pd.DataFrame.from_dict(metadata)  # type: ignore
        if "rlnImageName" not in particle_dataframe_at_index.keys():
            raise IOError(
                "Tried to read STAR file for "
                f"`RelionParticleStackDataset` index = {index}, "
                "but no entry found for 'rlnImageName'."
            )
        # ... the following line is necessary for the image dataset to work with both the
        # helical dataset and the regular dataset
        particle_index = np.asarray(particle_dataframe_at_index.index, dtype=int)
        # ... then, load stack of images
        images = _load_image_stack_from_mrc(
            particle_index,
            particle_dataframe_at_index,
            self.path_to_relion_project,
        )
        if parameters.pose.offset_x_in_angstroms.ndim == 0:
            images = jnp.squeeze(images)

        # ... reset boolean
        self.parameter_dataset.loads_metadata = loads_metadata
        if not loads_metadata:
            parameters = RelionParticleParameters(
                parameters.instrument_config, parameters.pose, parameters.transfer_theory
            )

        return RelionParticleStack(parameters, images)

    @override
    def __len__(self) -> int:
        return len(self.parameter_dataset)

    @override
    def __setitem__(
        self,
        index: int | slice | Int[np.ndarray, ""],
        value: RelionParticleStack | Float[NDArrayLike, "... _ _"],
    ):
        if isinstance(index, Int[np.ndarray, "_"]):  # type: ignore
            raise ValueError(
                "When setting `particle_stack_dataset[index] = ...`, "
                "it is not supported to pass `index` as a 1D numpy-array."
            )
        if isinstance(value, RelionParticleStack):
            self.parameter_dataset[index] = value.parameters
            images, parameters = np.asarray(value.images), value.parameters
        elif isinstance(value, NDArrayLike):  # type: ignore
            images, parameters = np.asarray(value), None
        else:
            raise ValueError(
                "Dataset entries can only be set with "
                "the `RelionParticleStack` type or array "
                "types, i.e. image arrays with shape "
                "`(n_images, dim, dim)` or `(dim, dim)`, where "
                "`dim` is equal to the 'rlnImageSize'."
            )
        n_particles = len(self.parameter_dataset)
        index_array = np.atleast_1d(_index_to_array(index, n_particles))
        self._write_images_at_index(index_array, images, parameters=parameters)

    @override
    def append(self, value: RelionParticleStack):
        """Add an entry or entries to the dataset.

        **Arguments:**

        - `value`:
            The `RelionParticleParameters` to add to the dataset.
        """
        start = len(self.parameter_dataset)
        parameters, images = value.parameters, value.images
        # Append parameters. This automatically sets the 'rlnImageName'
        # column to NaNs
        self.parameter_dataset.append(parameters)
        # Write images
        stop = len(self.parameter_dataset)
        index_array = np.arange(start, stop, dtype=int)
        self._write_images_at_index(index_array, images, parameters=parameters)

    @override
    def write_images(
        self,
        path_to_filename: str | pathlib.Path,
        images: Float[NDArrayLike, "... _ _"],
        pixel_size: float | Float[NDArrayLike, ""] = -1.0,
    ):
        if images.ndim == 2:
            images = np.atleast_3d(images)
        write_image_stack_to_mrc(images, pixel_size, path_to_filename)

    @property
    def path_to_relion_project(self) -> pathlib.Path:
        return self._path_to_relion_project

    @property
    def filename_settings(self) -> ImageFilenameSettings:
        return self._filename_settings

    @filename_settings.setter
    def filename_settings(self, value: dict[str, Any]):
        self._filename_settings = _dict_to_filename_settings(value)

    @property
    def parameter_dataset(self) -> AbstractRelionParticleParameterDataset:
        return self._parameter_dataset

    def _write_images_at_index(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: Optional[RelionParticleParameters] = None,
    ):
        # Get relevant metadata
        particle_data = self.parameter_dataset.starfile_data["particles"]
        optics_data = self.parameter_dataset.starfile_data["optics"]
        if parameters is None:
            optics_group = _get_optics_group_from_particle_data(
                particle_data.iloc[index_array], optics_data
            )
            pixel_size, dim = (
                float(optics_group["rlnImagePixelSize"]),
                int(optics_group["rlnImageSize"]),
            )
        else:
            pixel_size, dim = (
                float(np.atleast_1d(parameters.instrument_config.pixel_size)[0]),
                parameters.instrument_config.shape[0],
            )
        if not (images.ndim in [2, 3] and images.shape[-2:] == (dim, dim)):
            raise ValueError(
                "Image(s) must be of "
                "shape `(n_images, dim, dim)` or `(dim, dim)`. "
                f"Tried writing image(s) of "
                f"shape {images.shape}."
            )
        # Prepare to write images
        images = np.atleast_3d(images)
        n_images, image_dim = images.shape[0], images.shape[1]
        n_particles = len(self.parameter_dataset)
        if dim != image_dim:
            raise ValueError(
                "Found inconsistent image shape and "
                "'rlnImageSize' entry. The image dimension "
                f"was {dim}, while the 'rlnImageSize' was "
                f"{image_dim}."
            )
        # Convert index into 1D ascending numpy array
        n_indices = index_array.size
        if n_images != n_indices:
            raise ValueError(
                "Tried to set dataset elements with an inconsistent number "
                f"of images. Found that the number of images was {n_images}, "
                f"while the number of dataset indices was {n_indices}."
            )
        # Get absolute path to the filename, as well as the 'rlnImageName'
        # column
        path_to_filename, rln_image_names = _make_image_filename(
            index_array,
            particle_data,
            n_particles,
            self.filename_settings,
            self.path_to_relion_project,
        )
        # Set the STAR file column
        particle_data["rlnImageName"].iloc[index_array] = rln_image_names
        self.parameter_dataset.starfile_data = dict(
            particles=particle_data, optics=optics_data
        )
        # ... and write the images to disk
        self.write_images(path_to_filename, images, pixel_size)


class RelionHelicalParameterDataset(AbstractRelionParticleParameterDataset):
    """Similar to a `RelionParticleParameterDataset`, but reads helical tubes.

    In particular, a `RelionHelicalParameterDataset` indexes one
    helical filament at a time. For example, after manual
    particle picking in RELION, we can index a particular filament
    with

    ```python
    # Read in a STAR file particle stack
    dataset = RelionHelicalParameterDataset(...)
    # ... get a particle stack for a filament
    parameters_for_a_filament = dataset[0]
    # ... get a particle stack for another filament
    parameters_for_another_filament = dataset[1]
    ```

    Unlike a `RelionParticleParameterDataset`, a `RelionHelicalParameterDataset`
    does not support fancy indexing.
    """

    def __init__(
        self,
        parameter_dataset: RelionParticleParameterDataset,
    ):
        """**Arguments:**

        - `parameter_dataset`:
            The wrappped `RelionParticleParameterDataset`. This will be
            slightly modified to read one helix at a time, rather than
            one image crop at a time.
        """
        # Validate the STAR file and store the dataset
        _validate_helical_starfile_data(parameter_dataset.starfile_data)
        self._parameter_dataset = parameter_dataset
        # Compute and store the number of filaments, number of filaments per micrograph
        # and micrograph names
        n_filaments_per_micrograph, micrograph_names = (
            _get_number_of_filaments_per_micrograph_in_helical_starfile_data(
                parameter_dataset.starfile_data
            )
        )
        self._n_filaments = int(np.sum(n_filaments_per_micrograph))
        self._n_filaments_per_micrograph = n_filaments_per_micrograph
        self._micrograph_names = micrograph_names

    def __getitem__(self, index: int | Int[np.ndarray, ""]) -> RelionParticleParameters:
        _validate_helical_dataset_index(type(self), index, len(self))
        # Get the particle stack indices corresponding to this filament
        particle_dataframe = self._parameter_dataset.starfile_data["particles"]
        particle_indices_at_filament_index = _get_particle_indices_at_filament_index(
            particle_dataframe,
            index,
            self._n_filaments_per_micrograph,
            self._micrograph_names,
        )
        # Access the particle stack at these particle indices
        return self._parameter_dataset[particle_indices_at_filament_index]

    def __len__(self) -> int:
        return self._n_filaments

    @override
    def __setitem__(
        self, index: int | Int[np.ndarray, ""], value: RelionParticleParameters
    ):
        raise NotImplementedError

    @override
    def append(self, value: RelionParticleParameters):
        raise NotImplementedError

    @override
    def save(self, **kwargs: Any):
        return self._parameter_dataset.save(**kwargs)

    @property
    @override
    def starfile_data(self) -> StarfileData:
        return self._parameter_dataset._starfile_data

    @starfile_data.setter
    @override
    def starfile_data(self, value: dict[str, pd.DataFrame]):
        self._parameter_dataset.starfile_data = value

    @property
    @override
    def loads_metadata(self) -> bool:
        return self._parameter_dataset._loads_metadata

    @loads_metadata.setter
    @override
    def loads_metadata(self, value: bool):
        self._parameter_dataset._loads_metadata = value

    @property
    @override
    def loads_envelope(self) -> bool:
        return self._parameter_dataset._loads_envelope

    @loads_envelope.setter
    @override
    def loads_envelope(self, value: bool):
        self._parameter_dataset._loads_envelope = value

    @property
    @override
    def broadcasts_optics_group(self) -> bool:
        return self._parameter_dataset._broadcasts_optics_group

    @broadcasts_optics_group.setter
    @override
    def broadcasts_optics_group(self, value: bool):
        self._parameter_dataset._broadcasts_optics_group = value

    @property
    @override
    def updates_optics_group(self) -> bool:
        return self._parameter_dataset._updates_optics_group

    @updates_optics_group.setter
    @override
    def updates_optics_group(self, value: bool):
        self._parameter_dataset._updates_optics_group = value


def _load_starfile_data(
    path_to_starfile: pathlib.Path, mode: Literal["r", "w"], overwrite: bool
) -> StarfileData:
    if mode == "r":
        if path_to_starfile.exists():
            starfile_data = read_and_validate_starfile(path_to_starfile)
            _validate_starfile_data(starfile_data)
        else:
            raise FileNotFoundError(
                f"Set `mode = '{mode}'`, but STAR file {str(path_to_starfile)} does not "
                "exist. To write a new STAR file, set `mode = 'w'`."
            )
    elif mode == "w":
        if path_to_starfile.exists() and not overwrite:
            raise FileExistsError(
                f"Set `mode = 'w'`, but STAR file {str(path_to_starfile)} already "
                "exists. To read an existing STAR file, set `mode = 'r'` or "
                "to erase an existing STAR file, set `mode = 'w'` and "
                "`overwrite=True`."
            )
        else:
            starfile_data = dict(
                optics=pd.DataFrame(columns=RELION_REQUIRED_OPTICS_KEYS),
                particles=pd.DataFrame(
                    columns=[*RELION_REQUIRED_PARTICLE_KEYS, *RELION_POSE_PARTICLE_KEYS]
                ),
            )
    else:
        raise ValueError(
            f"Passed unsupported `mode = {mode}`. Supported modes are 'r' and 'w'."
        )
    return StarfileData(
        particles=starfile_data["particles"], optics=starfile_data["optics"]
    )


#
# STAR file reading
#
def _make_pytrees_from_starfile(
    starfile_dataframe,
    optics_group,
    broadcasts_optics_group,
    loads_envelope,
    make_config_fn,
) -> tuple[InstrumentConfig, ContrastTransferTheory, EulerAnglePose]:
    defocus_in_angstroms = (
        jnp.asarray(starfile_dataframe["rlnDefocusU"])
        + jnp.asarray(starfile_dataframe["rlnDefocusV"])
    ) / 2
    astigmatism_in_angstroms = jnp.asarray(
        starfile_dataframe["rlnDefocusU"]
    ) - jnp.asarray(starfile_dataframe["rlnDefocusV"])
    astigmatism_angle = jnp.asarray(starfile_dataframe["rlnDefocusAngle"])
    phase_shift = jnp.asarray(starfile_dataframe["rlnPhaseShift"])
    # ... optics group data
    image_size = int(optics_group["rlnImageSize"])
    pixel_size = jnp.asarray(optics_group["rlnImagePixelSize"])
    voltage_in_kilovolts = jnp.asarray(optics_group["rlnVoltage"])
    spherical_aberration_in_mm = jnp.asarray(optics_group["rlnSphericalAberration"])
    amplitude_contrast_ratio = jnp.asarray(optics_group["rlnAmplitudeContrast"])
    # ... create cryojax objects. First, the InstrumentConfig
    image_shape = (image_size, image_size)
    batch_dim = 0 if defocus_in_angstroms.ndim == 0 else defocus_in_angstroms.shape[0]
    instrument_config = _make_config(
        image_shape,
        pixel_size,
        voltage_in_kilovolts,
        batch_dim,
        make_config_fn,
        broadcasts_optics_group,
    )
    # ... now the ContrastTransferTheory
    if loads_envelope:
        b_factor, scale_factor = (
            (
                jnp.asarray(starfile_dataframe["rlnCtfBfactor"])
                if "rlnCtfBfactor" in starfile_dataframe.keys()
                else None
            ),
            (
                jnp.asarray(starfile_dataframe["rlnCtfScalefactor"])
                if "rlnCtfScalefactor" in starfile_dataframe.keys()
                else None
            ),
        )
        envelope = _make_envelope_function(scale_factor, b_factor)
    else:
        envelope = None

    transfer_theory = _make_transfer_theory(
        defocus_in_angstroms,
        astigmatism_in_angstroms,
        astigmatism_angle,
        spherical_aberration_in_mm,
        amplitude_contrast_ratio,
        phase_shift,
        envelope,
    )
    # ... and finally, the EulerAnglePose
    pose = EulerAnglePose()
    # ... values for the pose are optional, so look to see if
    # each key is present
    particle_keys = starfile_dataframe.keys()
    # Read the pose. first, xy offsets
    rln_origin_x_angst = (
        starfile_dataframe["rlnOriginXAngst"]
        if "rlnOriginXAngst" in particle_keys
        else 0.0
    )
    rln_origin_y_angst = (
        starfile_dataframe["rlnOriginYAngst"]
        if "rlnOriginYAngst" in particle_keys
        else 0.0
    )
    # ... rot angle
    rln_angle_rot = (
        starfile_dataframe["rlnAngleRot"] if "rlnAngleRot" in particle_keys else 0.0
    )
    # ... tilt angle
    if "rlnAngleTilt" in particle_keys:
        rln_angle_tilt = starfile_dataframe["rlnAngleTilt"]
    elif "rlnAngleTiltPrior" in particle_keys:  # support for helices
        rln_angle_tilt = starfile_dataframe["rlnAngleTiltPrior"]
    else:
        rln_angle_tilt = 0.0
    # ... psi angle
    if "rlnAnglePsi" in particle_keys:
        # Relion uses -999.0 as a placeholder for an un-estimated in-plane
        # rotation
        if isinstance(starfile_dataframe["rlnAnglePsi"], pd.Series):
            # ... check if all values are equal to -999.0. If so, just
            # replace the whole pandas.Series with 0.0
            if (
                starfile_dataframe["rlnAnglePsi"].nunique() == 1
                and starfile_dataframe["rlnAnglePsi"].iloc[0] == -999.0
            ):
                rln_angle_psi = 0.0
            # ... otherwise, replace -999.0 values with 0.0
            else:
                rln_angle_psi = starfile_dataframe["rlnAnglePsi"].where(
                    lambda x: x != -999.0, 0.0
                )
        else:
            # ... if the column is just equal to a float, then
            # directly check if it is equal to -999.0
            rln_angle_psi = (
                0.0
                if starfile_dataframe["rlnAnglePsi"] == -999.0
                else starfile_dataframe["rlnAnglePsi"]
            )
    elif "rlnAnglePsiPrior" in particle_keys:  # support for helices
        rln_angle_psi = starfile_dataframe["rlnAnglePsiPrior"]
    else:
        rln_angle_psi = 0.0
    # Now, flip the sign of the translations and transpose rotations.
    # RELION's convention thinks about the translation as "undoing" the translation
    # and rotation in the image
    pose_parameter_names = (
        "offset_x_in_angstroms",
        "offset_y_in_angstroms",
        "phi_angle",
        "theta_angle",
        "psi_angle",
    )
    pose_parameter_values = (
        -rln_origin_x_angst,
        -rln_origin_y_angst,
        -rln_angle_rot,
        -rln_angle_tilt,
        -rln_angle_psi,
    )
    # ... fill the EulerAnglePose will keys that are present. if they are not
    # present, keep the default values in the `pose = EulerAnglePose()`
    # instantiation
    maybe_make_full = lambda param: (
        np.full((batch_dim,), param)
        if batch_dim > 0 and np.asarray(param).shape == ()
        else param
    )
    pose = eqx.tree_at(
        lambda p: tuple([getattr(p, name) for name in pose_parameter_names]),
        pose,
        tuple([jnp.asarray(maybe_make_full(value)) for value in pose_parameter_values]),
    )

    return instrument_config, transfer_theory, pose


def _make_config(
    image_shape,
    pixel_size,
    voltage_in_kilovolts,
    batch_dim,
    make_config_fn,
    broadcasts_optics_group,
):
    make_fn = lambda ps, volt: make_config_fn(image_shape, ps, volt)
    if broadcasts_optics_group:
        make_fn_vmap = eqx.filter_vmap(make_fn)
        return (
            make_fn(pixel_size, voltage_in_kilovolts)
            if batch_dim == 0
            else make_fn_vmap(
                jnp.full((batch_dim,), pixel_size),
                jnp.full((batch_dim,), voltage_in_kilovolts),
            )
        )
    else:
        return make_fn(pixel_size, voltage_in_kilovolts)


def _make_envelope_function(amp, b_factor):
    if b_factor is None and amp is None:
        warnings.warn(
            "loads_envelope was set to True, but no envelope parameters were found. "
            + "Setting envelope as None. "
            + "Make sure your starfile is correctly formatted or set loads_envelope=False"
        )
        return None

    elif b_factor is None and amp is not None:

        def _make_const_env(amp):
            return Constant(amp)

        @eqx.filter_vmap(in_axes=0, out_axes=0)
        def _make_const_env_vmap(amp):
            return _make_const_env(amp)

        return _make_const_env(amp) if amp.ndim == 0 else _make_const_env_vmap(amp)
    else:
        if amp is None:
            amp = jnp.asarray(1.0) if b_factor.ndim == 0 else jnp.ones_like(b_factor)

        def _make_gaussian_env(amp, b):
            return FourierGaussian(amplitude=amp, b_factor=b)

        @eqx.filter_vmap(in_axes=(0, 0), out_axes=0)
        def _make_gaussian_env_vmap(amp, b):
            return _make_gaussian_env(amp, b)

        return (
            _make_gaussian_env(amp, b_factor)
            if b_factor.ndim == 0
            else _make_gaussian_env_vmap(amp, b_factor)
        )


def _make_transfer_theory(defocus, astig, angle, sph, ac, ps, env=None):
    if env is not None:

        def _make_w_env(defocus, astig, angle, sph, ac, ps, env):
            ctf = AberratedAstigmaticCTF(
                defocus_in_angstroms=defocus,
                astigmatism_in_angstroms=astig,
                astigmatism_angle=angle,
                spherical_aberration_in_mm=sph,
            )
            return ContrastTransferTheory(
                ctf, env, amplitude_contrast_ratio=ac, phase_shift=ps
            )

        @eqx.filter_vmap(in_axes=(0, 0, 0, None, None, 0, 0), out_axes=0)
        def _make_w_env_vmap(defocus, astig, angle, sph, ac, ps, env):
            return _make_w_env(defocus, astig, angle, sph, ac, ps, env)

        return (
            _make_w_env(defocus, astig, angle, sph, ac, ps, env)
            if defocus.ndim == 0
            else _make_w_env_vmap(defocus, astig, angle, sph, ac, ps, env)
        )

    else:

        def _make_wo_env(defocus, astig, angle, sph, ac, ps):
            ctf = AberratedAstigmaticCTF(
                defocus_in_angstroms=defocus,
                astigmatism_in_angstroms=astig,
                astigmatism_angle=angle,
                spherical_aberration_in_mm=sph,
            )
            return ContrastTransferTheory(
                ctf, envelope=None, amplitude_contrast_ratio=ac, phase_shift=ps
            )

        @eqx.filter_vmap(in_axes=(0, 0, 0, None, None, 0), out_axes=0)
        def _make_wo_env_vmap(defocus, astig, angle, sph, ac, ps):
            return _make_wo_env(defocus, astig, angle, sph, ac, ps)

        return (
            _make_wo_env(defocus, astig, angle, sph, ac, ps)
            if defocus.ndim == 0
            else _make_wo_env_vmap(defocus, astig, angle, sph, ac, ps)
        )


def _load_image_stack_from_mrc(
    index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"],
    particle_dataframe: pd.DataFrame,
    path_to_relion_project: str | pathlib.Path,
) -> Float[Array, "... y_dim x_dim"]:
    # Load particle image stack rlnImageName
    image_stack_index_and_name = particle_dataframe["rlnImageName"]

    if all([isinstance(elem, str) for elem in image_stack_index_and_name]):
        # ... split the pandas.Series into a pandas.DataFrame with two columns:
        # one for the image index and another for the filename
        image_stack_index_and_name_dataframe = (
            image_stack_index_and_name.str.split("@", expand=True)
        ).reset_index()
        # ... check dtype and shape of images
        path_to_test_image_stack = pathlib.Path(
            path_to_relion_project,
            np.asarray(image_stack_index_and_name_dataframe[1], dtype=object)[0],
        )
        with mrcfile.mmap(path_to_test_image_stack, mode="r", permissive=True) as mrc:
            mrc_data = np.asarray(mrc.data)
            test_image = mrc_data if mrc_data.ndim == 2 else mrc_data[0]
            image_dtype, image_shape = test_image.dtype, test_image.shape
        # ... allocate memory for stack
        n_images = len(image_stack_index_and_name_dataframe)
        image_stack = np.empty((n_images, *image_shape), dtype=image_dtype)
        # ... get unique mrc files
        unique_mrc_files = image_stack_index_and_name_dataframe[1].unique()
        # ... load images to image_stack
        for unique_mrc in unique_mrc_files:
            # ... get the indices for this particular mrc file
            indices_in_mrc = image_stack_index_and_name_dataframe[1] == unique_mrc
            # ... relion convention starts indexing at 1, not 0
            filtered_df = image_stack_index_and_name_dataframe[indices_in_mrc]
            particle_index = filtered_df[0].values.astype(int) - 1
            with mrcfile.mmap(
                pathlib.Path(path_to_relion_project, unique_mrc),
                mode="r",
                permissive=True,
            ) as mrc:
                mrc_data = np.asarray(mrc.data)
                if mrc_data.ndim == 2:
                    image_stack[filtered_df.index] = mrc_data
                else:
                    image_stack[filtered_df.index] = mrc_data[particle_index]

    else:
        raise IOError(
            "Error reading image(s) in STAR file for "
            f"`RelionParticleStackDataset` index = {index}."
        )

    return jnp.asarray(image_stack)


def _get_particle_indices_at_filament_index(
    particle_dataframe,
    filament_index,
    n_filaments_per_micrograph,
    micrograph_names,
):
    # ... map the filament index to a micrograph index
    n_filaments_per_micrograph = np.asarray(n_filaments_per_micrograph, dtype=int)
    last_index_of_filament_per_micrograph = np.cumsum(n_filaments_per_micrograph) - 1
    micrograph_index = np.where(last_index_of_filament_per_micrograph >= filament_index)[
        0
    ].min()
    # Get the filament index in this particular micrograph
    filament_index_in_micrograph = (n_filaments_per_micrograph[micrograph_index] - 1) - (
        last_index_of_filament_per_micrograph[micrograph_index] - filament_index
    )
    # .. get the data blocks only at the filament corresponding to the filament index
    particle_dataframe_at_micrograph = particle_dataframe[
        particle_dataframe["rlnMicrographName"] == micrograph_names[micrograph_index]
    ]
    particle_dataframe_at_filament = particle_dataframe_at_micrograph[
        particle_dataframe_at_micrograph["rlnHelicalTubeID"]
        == filament_index_in_micrograph + 1
    ]
    return np.asarray(particle_dataframe_at_filament.index, dtype=int)


def _get_number_of_filaments_per_micrograph_in_helical_starfile_data(
    starfile_data: StarfileData,
) -> tuple[list[int], list[str]]:
    particle_dataframe = starfile_data["particles"]
    micrograph_names = particle_dataframe["rlnMicrographName"].unique().tolist()
    n_filaments_per_micrograph = list(
        int(
            particle_dataframe[
                particle_dataframe["rlnMicrographName"] == micrograph_name
            ]["rlnHelicalTubeID"].max()
        )
        for micrograph_name in micrograph_names
    )

    return n_filaments_per_micrograph, micrograph_names  # type: ignore


def _validate_dataset_index(cls, index, n_rows):
    index_error_msg = lambda idx: (
        f"The index at which the `{cls.__name__}` was accessed was out of bounds! "
        f"The number of rows in the dataset is {n_rows}, but you tried to "
        f"access the index {idx}."
    )
    # ... pandas has bad error messages for its indexing
    if isinstance(index, (int, np.integer)):  # type: ignore
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
            f"`{cls.__name__}`. Indexing by integers is supported, one-dimensional "
            "fancy indexing is supported, and numpy-array indexing is supported. "
            "For example, like `particle = particle_dataset[0]`, "
            "`particle_stack = particle_dataset[0:5]`, "
            "or `particle_stack = dataset[np.array([1, 4, 3, 2])]`."
        )


def _validate_helical_dataset_index(cls, filament_index, n_filaments):
    if not isinstance(filament_index, (int, np.integer)):  # type: ignore
        raise IndexError(
            f"When indexing a `{cls.__name__}`, only "
            f"python or numpy-like integer particle_index are supported, such as "
            "`helical_particle_stack = helical_dataset[3]`. "
            f"Got index {filament_index} of type {type(filament_index)}."
        )
    # Make sure the filament index is in-bounds
    if filament_index + 1 > n_filaments:
        raise IndexError(
            f"The index at which the `{cls.__name__}` was "
            f"accessed was out of bounds! The number of filaments in "
            f"the dataset is {n_filaments}, but you tried to "
            f"access the index {filament_index}."
        )


def _validate_starfile_data(starfile_data: dict[str, pd.DataFrame]):
    if "particles" not in starfile_data.keys():
        raise ValueError("Missing key 'particles' in `starfile.read` output.")
    else:
        if not set(RELION_REQUIRED_PARTICLE_KEYS).issubset(
            set(starfile_data["particles"].keys())
        ):
            raise ValueError(
                "Missing required keys in starfile 'particles' group. "
                f"Required keys are {RELION_REQUIRED_PARTICLE_KEYS}."
            )
    if "optics" not in starfile_data.keys():
        raise ValueError("Missing key 'optics' in `starfile.read` output.")
    else:
        if not set(RELION_REQUIRED_OPTICS_KEYS).issubset(
            set(starfile_data["optics"].keys())
        ):
            raise ValueError(
                "Missing required keys in starfile 'optics' group. "
                f"Required keys are {RELION_REQUIRED_OPTICS_KEYS}."
            )


def _validate_helical_starfile_data(starfile_data: StarfileData):
    particle_dataframe = starfile_data["particles"]
    if "rlnHelicalTubeID" not in particle_dataframe.columns:
        raise ValueError(
            "Missing column 'rlnHelicalTubeID' in `starfile.read` output. "
            "This column must be present when using a "
            "`RelionHelicalParameterDataset`."
        )


#
# STAR file writing. First, functions for writing parameters
#
def _params_to_optics_data(
    parameters: RelionParticleParameters, optics_group_index: int
) -> pd.DataFrame:
    shape = parameters.instrument_config.shape
    if shape[0] == shape[1]:
        dim = shape[0]
    else:
        raise ValueError(
            "Found non-square shape in "
            "`RelionParticleParameters.instrument_config.shape`. Only "
            "square shapes are supported."
        )
    pixel_size = parameters.instrument_config.pixel_size
    voltage_in_kilovolts = parameters.instrument_config.voltage_in_kilovolts
    amplitude_contrast_ratio = parameters.transfer_theory.amplitude_contrast_ratio
    if isinstance(parameters.transfer_theory.ctf, AberratedAstigmaticCTF):
        spherical_aberration_in_mm = getattr(
            parameters.transfer_theory.ctf, "spherical_aberration_in_mm"
        )
    else:
        raise ValueError(
            "`RelionParticleParameters.transfer_theory.ctf` must be type "
            "`AberratedAstigmaticCTF`. Got type "
            f"{type(parameters.transfer_theory.ctf).__name__}."
        )
    optics_group_dict = {
        "rlnOpticsGroup": optics_group_index,
        "rlnImageSize": dim,
        "rlnImagePixelSize": pixel_size,
        "rlnVoltage": voltage_in_kilovolts,
        "rlnSphericalAberration": spherical_aberration_in_mm,
        "rlnAmplitudeContrast": amplitude_contrast_ratio,
    }
    for k, v in optics_group_dict.items():
        if isinstance(v, Array | np.ndarray):
            arr = np.atleast_1d(np.asarray(v))
            if arr.size > 1:
                if np.unique(arr).size > 1:
                    raise ValueError(
                        "Tried to fill a RELION optics group entry with an array "
                        "that has multiple unique values. Optics group compatible "
                        "arrays in `RelionParticleParameters`, such as "
                        "`RelionParticleParameters.instrument_config.pixel_size`, "
                        "must be either scalars or arrays all with the same value. "
                        f"Error occurred when filling '{k}' with array {v}."
                    )
            optics_group_dict[k] = arr.ravel()[0, None]
        else:
            optics_group_dict[k] = [v]

    return pd.DataFrame.from_dict(optics_group_dict)


def _params_to_particle_data(
    parameters: RelionParticleParameters,
    optics_group_index: Optional[int] = None,
) -> pd.DataFrame:
    particles_dict = {}
    # Fill CTF parameters
    transfer_theory = parameters.transfer_theory
    if isinstance(transfer_theory.ctf, AberratedAstigmaticCTF):
        particles_dict["rlnDefocusU"] = (
            transfer_theory.ctf.defocus_in_angstroms
            + transfer_theory.ctf.astigmatism_in_angstroms / 2
        )
        particles_dict["rlnDefocusV"] = (
            transfer_theory.ctf.defocus_in_angstroms
            - transfer_theory.ctf.astigmatism_in_angstroms / 2
        )
        particles_dict["rlnDefocusAngle"] = transfer_theory.ctf.astigmatism_angle
    else:
        raise ValueError(
            "`RelionParticleParameters.transfer_theory.ctf` must be type "
            "`AberratedAstigmaticCTF`. Got type "
            f"{type(transfer_theory.ctf).__name__}."
        )

    if isinstance(transfer_theory.envelope, FourierGaussian):
        particles_dict["rlnCtfBfactor"] = transfer_theory.envelope.b_factor
        particles_dict["rlnCtfScalefactor"] = transfer_theory.envelope.amplitude
    elif isinstance(transfer_theory.envelope, Constant):
        particles_dict["rlnCtfScalefactor"] = transfer_theory.envelope.value
    elif transfer_theory.envelope is None:
        pass
    else:
        raise ValueError(
            "`RelionParticleParameters.transfer_theory.envelope` must "
            "either be type `cryojax.image.operators.FourierGaussian` "
            "or `cryojax.image.operators.Constant`. Got "
            f"{type(transfer_theory.envelope).__name__}."
        )
    particles_dict["rlnPhaseShift"] = transfer_theory.phase_shift
    # Now, pose parameters
    pose = parameters.pose
    if not isinstance(pose, EulerAnglePose):
        raise ValueError(
            "`RelionParticleParameters.pose` must be type "
            "`EulerAnglePose`. Got type "
            f"{type(pose).__name__}."
        )
    particles_dict["rlnOriginXAngst"] = -pose.offset_x_in_angstroms
    particles_dict["rlnOriginYAngst"] = -pose.offset_y_in_angstroms
    particles_dict["rlnAngleRot"] = -pose.phi_angle
    particles_dict["rlnAngleTilt"] = -pose.theta_angle
    particles_dict["rlnAnglePsi"] = -pose.psi_angle
    # Now, broadcast parameters to same dimension
    n_particles = pose.offset_x_in_angstroms.size
    for k, v in particles_dict.items():
        if v.shape == ():
            particles_dict[k] = np.full((n_particles,), np.asarray(v))
        elif v.size == n_particles:
            particles_dict[k] = np.asarray(v.ravel())
        else:
            raise ValueError(
                "Found inconsistent number of particles "
                "in `RelionParticleParameters` instance. Arrays "
                "in this class must either be scalars or "
                "have the same number of dimensions."
            )
    # Now, miscellaneous parameters
    if optics_group_index is not None:
        particles_dict["rlnOpticsGroup"] = np.full(
            (n_particles,), optics_group_index, dtype=int
        )

    return pd.DataFrame.from_dict(particles_dict)


def _make_optics_group_index(optics_data: pd.DataFrame) -> int:
    optics_group_indices = np.asarray(optics_data["rlnOpticsGroup"], dtype=int)
    last_optics_group_index = (
        0 if optics_group_indices.size == 0 else int(optics_group_indices[-1])
    )
    return last_optics_group_index + 1


def _parse_optics_group_index(particle_data_at_index: pd.DataFrame | pd.Series) -> int:
    # ... read optics data
    optics_group_indices = np.unique(
        np.atleast_1d(np.asarray(particle_data_at_index["rlnOpticsGroup"]))
    )
    if optics_group_indices.size > 1:
        raise NotImplementedError(
            "Tried to read multiple particles at once that belong "
            "to different optics groups, but this is not yet "
            "implemented. In the meantime, try reading one particle "
            "at a time."
        )
    optics_group_index = optics_group_indices[0]

    return int(optics_group_index)


def _get_optics_group_from_index(
    optics_data: pd.DataFrame, optics_group_index: int
) -> pd.Series:
    return optics_data[optics_data["rlnOpticsGroup"] == optics_group_index].iloc[0]


def _get_optics_group_from_particle_data(
    particle_data_at_index: pd.DataFrame | pd.Series, optics_data: pd.DataFrame
) -> pd.Series:
    optics_group_index = _parse_optics_group_index(particle_data_at_index)
    return _get_optics_group_from_index(optics_data, optics_group_index)


#
# Now, functions for writing image files
#
def _set_empty_image_names(particle_data: pd.DataFrame):
    particle_data["rlnImageName"] = pd.Series(dtype=str)


def _parse_filename_for_number(path_to_filename: pathlib.Path) -> int:
    filename = path_to_filename.name
    match = re.search(r"[^0-9](\d+)\.[^.]+$", filename)
    try:
        file_number = int(match.group(1))  # type: ignore
    except Exception as err:
        raise IOError(
            f"Could not get the file number from file {str(path_to_filename)} "
            "Files must be enumerated with the trailing part of the "
            "filename as the file number, like so: '/path/to/file-0000.txt'. "
            f"When extracting the file number and converting it to an integer, "
            f"found error:\n\t{err}"
        )
    return file_number


def _format_number_for_filename(file_number: int, n_characters: int = 6):
    if file_number == 0:
        return "0" * n_characters
    else:
        n_digits = int(np.log10(file_number)) + 1
        return "0" * (n_characters - n_digits) + str(file_number)


def _dict_to_filename_settings(d: dict[str, Any]) -> ImageFilenameSettings:
    prefix = d["prefix"] if "prefix" in d else ""
    output_folder = d["output_folder"] if "output_folder" in d else ""
    delimiter = d["delimiter"] if "delimiter" in d else "_"
    n_characters = d["n_characters"] if "n_characters" in d else 6
    return ImageFilenameSettings(
        prefix=prefix,
        output_folder=output_folder,
        delimiter=delimiter,
        n_characters=n_characters,
    )


def _make_image_filename(
    index: Int[np.ndarray, " _"],
    particle_data: pd.DataFrame,
    n_particles: int,
    filename_settings: ImageFilenameSettings,
    path_to_relion_project: pathlib.Path,
) -> tuple[pathlib.Path, list[str]]:
    # Get the file number for this MRC file
    if n_particles == 0:
        file_number = 0
    else:
        last_index = index[0] - 1
        if last_index == -1:
            file_number = 0
        else:
            last_filename = particle_data["rlnImageName"].iloc[last_index]
            if pd.isna(last_filename):
                raise IOError(
                    "Tried to assign a number to the MRC file while writing "
                    "images, but could not grab the previous file number at "
                    f"index {int(last_index)}. At this index, found that the "
                    "filename was NaN."
                )
            else:
                file_number = _parse_filename_for_number(last_filename)
    # Unpack settings
    prefix = filename_settings["prefix"]
    output_folder = filename_settings["output_folder"]
    delimiter = filename_settings["delimiter"]
    n_characters = filename_settings["n_characters"]
    # Generate filename
    file_number_fmt = _format_number_for_filename(file_number, n_characters=n_characters)
    if prefix == "":
        relative_path_to_filename = str(
            pathlib.Path(output_folder, file_number_fmt + ".mrcs")
        )
    else:
        relative_path_to_filename = str(
            pathlib.Path(output_folder, prefix + delimiter + file_number_fmt + ".mrcs")
        )
    # Finally, generate the 'rln_image_name' column, which includes the particle index
    rln_image_names = [
        _format_number_for_filename(i, n_characters) + "@" + relative_path_to_filename
        for i in index
    ]
    # Finally, the path to the filename
    path_to_filename = pathlib.Path(path_to_relion_project, relative_path_to_filename)

    return path_to_filename, rln_image_names


def _index_to_array(indices: slice | int | np.ndarray, size: int) -> np.ndarray:
    if isinstance(indices, slice):
        return np.asarray(range(*indices.indices(size)))
    else:
        return np.asarray(indices, dtype=int)
