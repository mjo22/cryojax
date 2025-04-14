"""Cryojax compatibility with [RELION](https://relion.readthedocs.io/en/release-5.0/)."""

import abc
import pathlib
from typing import Any, Callable
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Int

from ...image.operators import FourierGaussian
from ...io import read_and_validate_starfile
from ...simulator import (
    AberratedAstigmaticCTF,
    ContrastTransferTheory,
    EulerAnglePose,
    InstrumentConfig,
)
from .._particle_data import (
    AbstractParticleParameterDataset,
    AbstractParticleStackDataset,
    ParticleStack,
)
from ._starfile_pytrees import RelionParticleParameters


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
    def path_to_relion_project(self) -> pathlib.Path:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def starfile_data(self) -> dict[str, pd.DataFrame]:
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


class RelionParticleParameterDataset(AbstractRelionParticleParameterDataset):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html)
    format.
    """

    def __init__(
        self,
        path_to_starfile: str | pathlib.Path,
        path_to_relion_project: str | pathlib.Path,
        *,
        loads_metadata: bool = False,
        broadcasts_optics_group: bool = True,
        loads_envelope: bool = False,
        make_config_fn: Callable[
            [tuple[int, int], Float[Array, "..."], Float[Array, "..."]],
            InstrumentConfig,
        ] = _default_make_config_fn,
    ):
        """**Arguments:**

        - `path_to_starfile`: The path to the Relion STAR file.
        - `path_to_relion_project`: The path to the Relion project directory.
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
        - `make_config_fn`:
            A function used for `InstrumentConfig` initialization that returns
            an `InstrumentConfig`. This is used to customize the metadata of the
            read object.
        """
        # Private attributes
        self._make_config_fn = make_config_fn
        # Properties without setters
        # ... read starfile and load path
        starfile_data = read_and_validate_starfile(path_to_starfile)
        _validate_starfile_data(starfile_data)
        self._path_to_relion_project = pathlib.Path(path_to_relion_project)
        self._starfile_data = starfile_data
        # Properties with setters
        self._loads_metadata = loads_metadata
        self._broadcasts_optics_group = broadcasts_optics_group
        self._loads_envelope = loads_envelope

    @override
    def __setitem__(self, index, value: RelionParticleParameters):
        raise NotImplementedError

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"]
    ) -> RelionParticleParameters:
        # Validate index
        n_rows = self.starfile_data["particles"].shape[0]
        _validate_dataset_index(type(self), index, n_rows)
        # ... read particle data
        particle_dataframe = self.starfile_data["particles"]
        particle_dataframe_at_index = particle_dataframe.iloc[index]
        # ... read optics data
        optics_group = self.starfile_data["optics"].iloc[0]
        # Load the image stack and STAR file parameters
        instrument_config, transfer_theory, pose = _make_pytrees_from_starfile(
            particle_dataframe_at_index,
            optics_group,
            self.broadcasts_optics_group,
            self.loads_envelope,
            self._make_config_fn,
        )
        # ... convert to dataframe for serialization
        if isinstance(particle_dataframe_at_index, pd.Series):
            particle_dataframe_at_index = pd.DataFrame(
                data=particle_dataframe_at_index.values[None, :],
                columns=particle_dataframe_at_index.index,
                index=[0],
            )
        return RelionParticleParameters(
            instrument_config,
            pose,
            transfer_theory,
            metadata=(
                particle_dataframe_at_index.to_dict() if self.loads_metadata else {}
            ),
        )

    @override
    def __len__(self) -> int:
        return len(self.starfile_data["particles"])

    @property
    @override
    def starfile_data(self) -> dict[str, pd.DataFrame]:
        return self._starfile_data

    @property
    @override
    def path_to_relion_project(self) -> pathlib.Path:
        return self._path_to_relion_project

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


class RelionParticleStackDataset(AbstractParticleStackDataset):
    """A dataset that wraps a RELION particle stack in
    [STAR](https://relion.readthedocs.io/en/latest/Reference/Conventions.html) format.
    """

    def __init__(self, param_dataset: AbstractRelionParticleParameterDataset):
        """**Arguments:**

        - `param_dataset`:
            The `RelionParticleParameterDataset`.
        """
        self._param_dataset = param_dataset

    @override
    def __setitem__(self, index, value: ParticleStack):
        raise NotImplementedError

    @override
    def __getitem__(
        self, index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"]
    ) -> ParticleStack:
        # ... make sure particle metadata is being loaded
        loads_metadata = self.param_dataset.loads_metadata
        self.param_dataset.loads_metadata = True
        # ... read parameters
        parameters = self.param_dataset[index]
        # ... and construct dataframe
        metadata = parameters.metadata
        particle_dataframe_at_index = pd.DataFrame.from_dict(metadata)  # type: ignore
        # ... the following line is necessary for the image dataset to work with both the
        # helical dataset and the regular dataset
        particle_index = np.asarray(particle_dataframe_at_index.index, dtype=int)
        # ... then, load stack of images
        images = _get_image_stack_from_mrc(
            particle_index,
            particle_dataframe_at_index,
            self.param_dataset.path_to_relion_project,
        )
        if parameters.pose.offset_x_in_angstroms.ndim == 0:
            images = jnp.squeeze(images)

        # ... reset boolean
        self.param_dataset.loads_metadata = loads_metadata
        if not loads_metadata:
            parameters = RelionParticleParameters(
                parameters.instrument_config, parameters.pose, parameters.transfer_theory
            )

        return ParticleStack(parameters, images)

    @override
    def __len__(self) -> int:
        return len(self.param_dataset)

    @property
    def param_dataset(self) -> AbstractRelionParticleParameterDataset:
        return self._param_dataset


class RelionHelicalParameterDataset(AbstractRelionParticleParameterDataset):
    """Similar to a `RelionParticleParameterDataset`, but reads helical tubes.

    In particular, a `RelionHelicalParameterDataset` indexes one
    helical filament at a time. For example, after manual
    particle picking in RELION, we can index a particular filament
    with

    ```python
    # Read in a STAR file particle stack
    helical_param_dataset = RelionHelicalParameterDataset(...)
    # ... get a particle stack for a filament
    params_for_a_filament = helical_particle_dataset[0]
    # ... get a particle stack for another filament
    params_for_another_filament = helical_particle_dataset[1]
    ```

    Unlike a `RelionParticleParameterDataset`, a `RelionHelicalParameterDataset`
    does not support fancy indexing.
    """

    def __init__(
        self,
        param_dataset: RelionParticleParameterDataset,
    ):
        """**Arguments:**

        - `param_dataset`:
            The wrappped `RelionParticleParameterDataset`. This will be
            slightly modified to read one helix at a time, rather than
            one image crop at a time.
        """
        # Validate the STAR file and store the dataset
        _validate_helical_starfile_data(param_dataset.starfile_data)
        self._param_dataset = param_dataset
        # Compute and store the number of filaments, number of filaments per micrograph
        # and micrograph names
        n_filaments_per_micrograph, micrograph_names = (
            _get_number_of_filaments_per_micrograph_in_helical_starfile_data(
                param_dataset.starfile_data
            )
        )
        self._n_filaments = int(np.sum(n_filaments_per_micrograph))
        self._n_filaments_per_micrograph = n_filaments_per_micrograph
        self._micrograph_names = micrograph_names

    @override
    def __setitem__(self, index, value: RelionParticleParameters):
        raise NotImplementedError

    def __getitem__(self, index: int | Int[np.ndarray, ""]) -> RelionParticleParameters:
        _validate_helical_dataset_index(type(self), index, len(self))
        # Get the particle stack indices corresponding to this filament
        particle_dataframe = self._param_dataset.starfile_data["particles"]
        particle_indices_at_filament_index = _get_particle_indices_at_filament_index(
            particle_dataframe,
            index,
            self._n_filaments_per_micrograph,
            self._micrograph_names,
        )
        # Access the particle stack at these particle indices
        return self._param_dataset[particle_indices_at_filament_index]

    def __len__(self) -> int:
        return self._n_filaments

    @property
    def starfile_data(self) -> dict[str, pd.DataFrame]:
        return self._param_dataset._starfile_data

    @property
    def path_to_relion_project(self) -> pathlib.Path:
        return self._param_dataset._path_to_relion_project

    @property
    @override
    def loads_metadata(self) -> bool:
        return self._param_dataset._loads_metadata

    @loads_metadata.setter
    @override
    def loads_metadata(self, value: bool):
        self._param_dataset._loads_metadata = value

    @property
    @override
    def loads_envelope(self) -> bool:
        return self._param_dataset._loads_envelope

    @loads_envelope.setter
    @override
    def loads_envelope(self, value: bool):
        self._param_dataset._loads_envelope = value

    @property
    @override
    def broadcasts_optics_group(self) -> bool:
        return self._param_dataset._broadcasts_optics_group

    @broadcasts_optics_group.setter
    @override
    def broadcasts_optics_group(self, value: bool):
        self._param_dataset._broadcasts_optics_group = value


def _make_pytrees_from_starfile(
    particle_blocks,
    optics_group,
    broadcasts_optics_group,
    loads_envelope,
    make_config_fn,
) -> tuple[InstrumentConfig, ContrastTransferTheory, EulerAnglePose]:
    defocus_in_angstroms = (
        jnp.asarray(particle_blocks["rlnDefocusU"])
        + jnp.asarray(particle_blocks["rlnDefocusV"])
    ) / 2
    astigmatism_in_angstroms = jnp.asarray(particle_blocks["rlnDefocusU"]) - jnp.asarray(
        particle_blocks["rlnDefocusV"]
    )
    astigmatism_angle = jnp.asarray(particle_blocks["rlnDefocusAngle"])
    phase_shift = jnp.asarray(particle_blocks["rlnPhaseShift"])
    # ... optics group data
    image_size = jnp.asarray(optics_group["rlnImageSize"])
    pixel_size = jnp.asarray(optics_group["rlnImagePixelSize"])
    voltage_in_kilovolts = float(optics_group["rlnVoltage"])  # type: ignore
    spherical_aberration_in_mm = jnp.asarray(optics_group["rlnSphericalAberration"])
    amplitude_contrast_ratio = jnp.asarray(optics_group["rlnAmplitudeContrast"])
    voltage_in_kilovolts = jnp.asarray(voltage_in_kilovolts)

    # ... create cryojax objects. First, the InstrumentConfig
    image_shape = (int(image_size), int(image_size))
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
                jnp.asarray(particle_blocks["rlnCtfBfactor"])
                if "rlnCtfBfactor" in particle_blocks.keys()
                else jnp.zeros_like(defocus_in_angstroms)
            ),
            (
                jnp.asarray(particle_blocks["rlnCtfScalefactor"])
                if "rlnCtfScalefactor" in particle_blocks.keys()
                else jnp.ones_like(defocus_in_angstroms)
            ),
        )
    else:
        b_factor, scale_factor = None, None
    transfer_theory = _make_transfer_theory(
        defocus_in_angstroms,
        astigmatism_in_angstroms,
        astigmatism_angle,
        spherical_aberration_in_mm,
        amplitude_contrast_ratio,
        phase_shift,
        scale_factor,
        b_factor,
    )
    # ... and finally, the EulerAnglePose
    pose = EulerAnglePose()
    # ... values for the pose are optional, so look to see if
    # each key is present
    particle_keys = particle_blocks.keys()
    pose_parameter_names_and_values = []
    if "rlnOriginXAngst" in particle_keys:
        pose_parameter_names_and_values.append(
            ("offset_x_in_angstroms", particle_blocks["rlnOriginXAngst"])
        )
    else:
        pose_parameter_names_and_values.append(("offset_x_in_angstroms", 0.0))
    if "rlnOriginYAngst" in particle_keys:
        pose_parameter_names_and_values.append(
            ("offset_y_in_angstroms", particle_blocks["rlnOriginYAngst"])
        )
    else:
        pose_parameter_names_and_values.append(("offset_y_in_angstroms", 0.0))
    if "rlnAngleRot" in particle_keys:
        pose_parameter_names_and_values.append(
            ("phi_angle", particle_blocks["rlnAngleRot"])
        )
    else:
        pose_parameter_names_and_values.append(("phi_angle", 0.0))
    if "rlnAngleTilt" in particle_keys:
        pose_parameter_names_and_values.append(
            ("theta_angle", particle_blocks["rlnAngleTilt"])
        )
    elif "rlnAngleTiltPrior" in particle_keys:  # support for helices
        pose_parameter_names_and_values.append(
            ("theta_angle", particle_blocks["rlnAngleTiltPrior"])
        )
    else:
        pose_parameter_names_and_values.append(("theta_angle", 0.0))
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
        pose_parameter_names_and_values.append(("psi_angle", particle_blocks_for_psi))
    elif "rlnAnglePsiPrior" in particle_keys:  # support for helices
        pose_parameter_names_and_values.append(
            ("psi_angle", particle_blocks["rlnAnglePsiPrior"])
        )
    else:
        pose_parameter_names_and_values.append(("psi_angle", 0.0))
    pose_parameter_names, pose_parameter_values = tuple(
        zip(*pose_parameter_names_and_values)
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


def _make_transfer_theory(defocus, astig, angle, sph, ac, ps, amp=None, b=None):
    if b is not None:

        def _make_w_env(defocus, astig, angle, sph, ac, ps, amp, b):
            ctf = AberratedAstigmaticCTF(
                defocus_in_angstroms=defocus,
                astigmatism_in_angstroms=astig,
                astigmatism_angle=angle,
                spherical_aberration_in_mm=sph,
            )
            envelope = FourierGaussian(b_factor=b, amplitude=amp)
            return ContrastTransferTheory(
                ctf, envelope, amplitude_contrast_ratio=ac, phase_shift=ps
            )

        @eqx.filter_vmap(in_axes=(0, 0, 0, None, None, 0, 0, 0), out_axes=0)
        def _make_w_env_vmap(defocus, astig, angle, sph, ac, ps, amp, b):
            return _make_w_env(defocus, astig, angle, sph, ac, ps, amp, b)

        return (
            _make_w_env(defocus, astig, angle, sph, ac, ps, amp, b)
            if defocus.ndim == 0
            else _make_w_env_vmap(defocus, astig, angle, sph, ac, ps, amp, b)
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


def _get_image_stack_from_mrc(
    index: int | slice | Int[np.ndarray, ""] | Int[np.ndarray, " N"],
    particle_dataframe,
    path_to_relion_project,
) -> Float[Array, "... y_dim x_dim"]:
    # Load particle image stack rlnImageName
    image_stack_index_and_name_series_or_str = particle_dataframe["rlnImageName"]
    if isinstance(image_stack_index_and_name_series_or_str, str):
        # In this block, the user most likely used standard indexing, like
        # `dataset = RelionParticleStackDataset(...); particle_stack = dataset[1]`
        image_stack_index_and_name_str = image_stack_index_and_name_series_or_str
        # ... split the whole string into its image index and filename
        relion_particle_index, image_stack_filename = (
            image_stack_index_and_name_str.split("@")
        )
        # ... create full path to the image stack
        path_to_image_stack = pathlib.Path(path_to_relion_project, image_stack_filename)
        # ... relion convention starts indexing at 1, not 0
        particle_index = np.asarray(relion_particle_index, dtype=int) - 1

        with mrcfile.mmap(path_to_image_stack, mode="r", permissive=True) as mrc:
            mrc_data = np.asarray(mrc.data)
            if mrc_data.ndim == 2:
                image_stack = mrc_data
            else:
                image_stack = mrc_data[particle_index]

    elif isinstance(image_stack_index_and_name_series_or_str, pd.Series):
        # In this block, the user most likely used fancy indexing, like
        # `dataset = RelionParticleStackDataset(...); particle_stack = dataset[1:10]`
        image_stack_index_and_name_series = image_stack_index_and_name_series_or_str
        # ... split the pandas.Series into a pandas.DataFrame with two columns:
        # one for the image index and another for the filename
        image_stack_index_and_name_dataframe = (
            image_stack_index_and_name_series.str.split("@", expand=True)
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
            "Could not read `rlnImageName` in STAR file for "
            f"`RelionParticleStackDataset` index equal to {index}."
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
    starfile_data: dict[str, pd.DataFrame],
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

    return n_filaments_per_micrograph, micrograph_names


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


def _validate_helical_starfile_data(starfile_data: dict[str, pd.DataFrame]):
    particle_dataframe = starfile_data["particles"]
    if "rlnHelicalTubeID" not in particle_dataframe.columns:
        raise ValueError(
            "Missing column 'rlnHelicalTubeID' in `starfile.read` output. "
            "This column must be present when using a "
            "`RelionHelicalParameterDataset`."
        )
