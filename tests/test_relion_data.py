"""
Check coverage with
pytest --cov-report term-missing:skip-covered --cov=src/cryojax/data/_relion tests/test_relion_data.py
"""  # noqa

import os
import shutil
from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest
from jaxtyping import TypeCheckError

import cryojax.simulator as cxs
from cryojax.data import (
    RelionParticleParameterFile,
    RelionParticleStackDataset,
    simulate_particle_stack,
)
from cryojax.data._particle_data.relion import (
    _validate_starfile_data,
)
from cryojax.image import operators as op
from cryojax.io import read_array_from_mrc
from cryojax.rotations import SO3


def compare_pytrees(pytree1, pytree2):
    arrays1, others1 = eqx.partition(pytree1, eqx.is_array)
    arrays2, others2 = eqx.partition(pytree2, eqx.is_array)

    bool_arrays = all(
        jax.tree.leaves(jax.tree.map(lambda x, y: jnp.allclose(x, y), arrays1, arrays2))
    )
    bool_others = all(
        jax.tree.leaves(jax.tree.map(lambda x, y: x == y, others1, others2))
    )
    return bool_arrays and bool_others


@pytest.fixture
def sample_starfile_path():
    return os.path.join(os.path.dirname(__file__), "data", "test_starfile.star")


@pytest.fixture
def sample_relion_project_path():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def sample_image_stack_path(sample_relion_project_path):
    return os.path.join(sample_relion_project_path, "000000.mrcs")


@pytest.fixture
def sample_image_stack(sample_image_stack_path):
    return read_array_from_mrc(sample_image_stack_path)


@pytest.fixture
def parameter_file(sample_starfile_path):
    return RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=True,
        loads_metadata=True,
    )


@pytest.fixture
def relion_parameters():
    instrument_config = cxs.InstrumentConfig(
        shape=(4, 4),
        pixel_size=1.5,
        voltage_in_kilovolts=300.0,
        padded_shape=(14, 14),
        pad_mode="constant",
    )

    pose = cxs.EulerAnglePose()
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.CTF(),
    )
    return dict(
        instrument_config=instrument_config, pose=pose, transfer_theory=transfer_theory
    )


#
# Tests for starfile loading
#
class TestErrorRaisingForLoading:
    def test_load_with_badparticle_name(self, parameter_file, sample_relion_project_path):
        parameter_file.starfile_data["particles"].loc[0, "rlnImageName"] = 0.0
        dataset = RelionParticleStackDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )
        with pytest.raises(TypeError):
            dataset[0]

        def test_load_with_badparticle_name2(
            self, parameter_file, sample_relion_project_path
        ):
            parameter_file.starfile_data["particles"].loc[0, "rlnImageName"] = "0000.mrcs"
            dataset = RelionParticleStackDataset(
                path_to_relion_project=sample_relion_project_path,
                parameter_file=parameter_file,
            )
            with pytest.raises(TypeError):
                dataset[0]

    def test_load_with_bad_shape(self, parameter_file, sample_relion_project_path):
        parameter_file.starfile_data["optics"].loc[0, "rlnImageSize"] = 1
        dataset = RelionParticleStackDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )
        with pytest.raises(ValueError):
            dataset[0]

    def test_with_bad_indices(self, parameter_file, sample_relion_project_path):
        dataset = RelionParticleStackDataset(
            path_to_relion_project=sample_relion_project_path,
            parameter_file=parameter_file,
        )

        # overflow index
        with pytest.raises(IndexError):
            parameter_file[len(parameter_file)]

        with pytest.raises(IndexError):
            dataset[len(dataset)]

        # overflow slice
        with pytest.raises(IndexError):
            parameter_file[len(parameter_file) :]

        with pytest.raises(IndexError):
            dataset[len(dataset) :]

        # wrong index type
        with pytest.raises(IndexError):
            parameter_file["wrong_index"]

        with pytest.raises(IndexError):
            dataset["wrong_index"]  # type: ignore

    def test_validate_starfile_data(self):
        with pytest.raises(ValueError):
            _validate_starfile_data({"wrong": pd.DataFrame({})})

        with pytest.raises(ValueError):
            _validate_starfile_data({"particles": pd.DataFrame({})})

        mock_particles_df = pd.DataFrame(
            {
                "rlnDefocusU": 0.0,
                "rlnDefocusV": 0.0,
                "rlnDefocusAngle": 0.0,
                "rlnPhaseShift": 0.0,
                "rlnImageName": "mock.mrcs",
            },
            index=[0],
        )
        with pytest.raises(ValueError):
            _validate_starfile_data({"particles": mock_particles_df})

        with pytest.raises(ValueError):
            _validate_starfile_data(
                {"particles": mock_particles_df, "optics": pd.DataFrame({})}
            )


def test_default_make_config_fn(sample_starfile_path):
    """Test the default make_config_fn function."""
    # Test with a valid input

    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=True,
        loads_metadata=True,
    )
    config = parameter_file[0]["instrument_config"]

    ref_config = cxs.InstrumentConfig(
        shape=(16, 16),
        pixel_size=12.0,
        voltage_in_kilovolts=300.0,
        padded_shape=(16, 16),
        pad_mode="constant",
    )

    assert config.shape == ref_config.shape
    assert config.pixel_size == ref_config.pixel_size
    assert config.voltage_in_kilovolts == ref_config.voltage_in_kilovolts
    assert (
        config.electrons_per_angstrom_squared == ref_config.electrons_per_angstrom_squared
    )

    assert config.padded_shape == ref_config.padded_shape
    assert config.pad_mode == ref_config.pad_mode


def test_load_starfile_envelope_params(sample_starfile_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=True,
        loads_metadata=True,
    )

    assert parameter_file.loads_envelope is True
    parameters = parameter_file[0]
    assert parameters["transfer_theory"].envelope is not None

    parameters = parameter_file[:]
    assert parameters["transfer_theory"].envelope is not None

    envelope = parameters["transfer_theory"].envelope
    # check that envelope params match
    for i in range(len(parameter_file)):
        # check b-factors
        np.testing.assert_allclose(
            envelope.b_factor[i],  # type: ignore
            parameters["metadata"]["rlnCtfBfactor"][i],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            envelope.amplitude[i],  # type: ignore
            parameters["metadata"]["rlnCtfScalefactor"][i],
            rtol=1e-5,
        )
    return


def test_load_starfile_ctf_params(sample_starfile_path):
    def compute_defocus(defU, defV):
        return 0.5 * (defU + defV)

    def compute_astigmatism(defU, defV):
        return defU - defV

    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=True,
    )

    assert parameter_file.loads_envelope is False

    parameters = parameter_file[0]
    assert parameters["transfer_theory"].envelope is None

    parameters = parameter_file[:]
    assert parameters["transfer_theory"].envelope is None

    transfer_theory = parameters["transfer_theory"]
    ctf = cast(cxs.AberratedAstigmaticCTF, transfer_theory.ctf)

    # check CTF parameters
    for i in range(len(parameter_file)):
        # defocus
        np.testing.assert_allclose(
            ctf.defocus_in_angstroms[i],
            compute_defocus(
                parameters["metadata"]["rlnDefocusU"][i],
                parameters["metadata"]["rlnDefocusV"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism
        np.testing.assert_allclose(
            ctf.astigmatism_in_angstroms[i],
            compute_astigmatism(
                parameters["metadata"]["rlnDefocusU"][i],
                parameters["metadata"]["rlnDefocusV"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism_angle
        np.testing.assert_allclose(
            ctf.astigmatism_angle[i],
            parameters["metadata"]["rlnDefocusAngle"][i],
            rtol=1e-5,
        )

        # phase shift
        np.testing.assert_allclose(
            transfer_theory.phase_shift[i],
            parameters["metadata"]["rlnPhaseShift"][i],
            rtol=1e-5,
        )

    return


def test_load_starfile_pose_params(sample_starfile_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=True,
    )

    parameters = parameter_file[:]
    pose = parameters["pose"]

    # check pose parameters
    for i in range(len(parameter_file)):
        # offset x
        np.testing.assert_allclose(
            pose.offset_x_in_angstroms[i],
            -parameters["metadata"]["rlnOriginXAngst"][i],  # conventions!
            rtol=1e-5,
        )

        # offset y
        np.testing.assert_allclose(
            pose.offset_y_in_angstroms[i],
            -parameters["metadata"]["rlnOriginYAngst"][i],  # conventions!
            rtol=1e-5,
        )

        # phi angle - AngleRot
        np.testing.assert_allclose(
            pose.phi_angle[i],
            -parameters["metadata"]["rlnAngleRot"][i],
            rtol=1e-5,
        )

        # theta angle - AngleTilt
        np.testing.assert_allclose(
            pose.theta_angle[i],
            -parameters["metadata"]["rlnAngleTilt"][i],
            rtol=1e-5,
        )

        # psi angle - AnglePsi
        np.testing.assert_allclose(
            pose.psi_angle[i],
            -parameters["metadata"]["rlnAnglePsi"][i],
            rtol=1e-5,
        )


def test_load_starfile_wo_metadata(sample_starfile_path):
    """Test loading a starfile without metadata."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=False,
    )

    # check that metadata is empty dict
    assert parameter_file[0]["metadata"] == {}
    assert parameter_file[:]["metadata"] == {}
    assert not parameter_file.loads_metadata


def test_load_optics_group_broadcasting(sample_starfile_path):
    """Test loading a starfile with optics group."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=True,
        broadcasts_optics_group=True,
    )

    parameters = parameter_file[:]
    instrument_config = parameters["instrument_config"]
    assert instrument_config.voltage_in_kilovolts.ndim > 0
    assert instrument_config.pixel_size.ndim > 0
    assert parameter_file.broadcasts_optics_group is True

    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=True,
        broadcasts_optics_group=False,
    )
    parameters = parameter_file[:]
    instrument_config = parameters["instrument_config"]
    assert instrument_config.voltage_in_kilovolts.ndim == 0
    assert instrument_config.pixel_size.ndim == 0
    assert parameter_file.broadcasts_optics_group is False

    return


def test_parameter_file_setters(sample_starfile_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=False,
        broadcasts_optics_group=False,
        updates_optics_group=False,
    )

    parameter_file.loads_metadata = True
    assert parameter_file.loads_metadata

    parameter_file.loads_envelope = True
    assert parameter_file.loads_envelope

    parameter_file.broadcasts_optics_group = True
    assert parameter_file.broadcasts_optics_group

    parameter_file.updates_optics_group = True
    assert parameter_file.updates_optics_group


def test_load_starfile_vs_mrcs_shape(sample_starfile_path, sample_relion_project_path):
    """Test loading a starfile with mrcs."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        loads_envelope=False,
        loads_metadata=False,
        broadcasts_optics_group=False,
    )
    dataset = RelionParticleStackDataset(parameter_file, sample_relion_project_path)

    particle_stack = dataset[:]
    instrument_config = particle_stack["parameters"]["instrument_config"]
    assert particle_stack["images"].shape == (
        len(parameter_file),
        *instrument_config.shape,
    )

    particle_stack = dataset[0]
    instrument_config = particle_stack["parameters"]["instrument_config"]
    assert particle_stack["images"].shape == instrument_config.shape

    particle_stack = dataset[0:2]
    instrument_config = particle_stack["parameters"]["instrument_config"]
    assert particle_stack["images"].shape == (2, *instrument_config.shape)

    assert len(dataset) == len(parameter_file)

    return


#
# Tests for starfile writing
#


@pytest.mark.parametrize(
    "index, loads_envelope",
    [
        (0, False),
        ([0, 1], False),
        (0, True),
    ],
)
def test_append_particle_parameters(index, loads_envelope):
    index = np.asarray(index)
    ndim = index.ndim

    @eqx.filter_vmap
    def make_particle_params(dummy_idx, metadata):
        instrument_config = cxs.InstrumentConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.CTF(),
            envelope=op.FourierGaussian() if loads_envelope else None,
        )
        return dict(
            instrument_config=instrument_config,
            pose=pose,
            transfer_theory=transfer_theory,
            metadata=metadata,
        )

    # Make particle parameters, using custom metadata
    metadata = pd.DataFrame(
        data={
            "rlnMicrographName": list("dummy/micrograph.mrc" for _ in range(index.size)),
            "rlnCoordinateX": np.atleast_1d(np.full_like(index, 2, dtype=int)),
            "rlnCoordinateY": np.atleast_1d(np.full_like(index, 1, dtype=int)),
        },
    )
    particle_params = make_particle_params(jnp.atleast_1d(index), metadata.to_dict())
    # ... custom metadata
    if ndim == 0:
        particle_params = jax.tree.map(
            lambda x: jnp.squeeze(x) if isinstance(x, jax.Array) else x, particle_params
        )
    # Add to dataset
    path_to_starfile = "tests/outputs/starfile_writing/test_particle_parameters.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exists_ok=True,
        loads_envelope=loads_envelope,
        loads_metadata=False,
    )
    parameter_file.append(particle_params)
    # Make sure parameters read and the same as what was appended
    loaded_particle_params = parameter_file[index]
    assert compare_pytrees(
        loaded_particle_params, eqx.tree_at(lambda x: x["metadata"], particle_params, {})
    )
    # Make sure custom metadata was added
    particle_dataframe = parameter_file.starfile_data["particles"]
    assert set(metadata.columns).issubset(particle_dataframe.columns)
    # Make sure dataframes are the same
    metadata_extracted = particle_dataframe.loc[
        particle_dataframe.index[np.atleast_1d(index)],
        ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"],
    ]
    np.testing.assert_equal(metadata.to_numpy(), metadata_extracted.to_numpy())


@pytest.mark.parametrize(
    "index, updates_optics_group, sets_envelope",
    [
        (0, False, False),
        ([0, 1], False, False),
        (0, True, False),
        (0, False, True),
    ],
)
def test_set_particle_parameters(
    sample_starfile_path,
    index,
    updates_optics_group,
    sets_envelope,
):
    index = np.asarray(index)
    n_particles, ndim = index.size, index.ndim

    def make_params(rng_key, metadata):
        rng_keys = jr.split(rng_key, n_particles)
        make_pose = eqx.filter_vmap(
            lambda rng_key: cxs.EulerAnglePose.from_rotation(SO3.sample_uniform(rng_key))
        )
        pose = make_pose(rng_keys)
        return dict(
            instrument_config=cxs.InstrumentConfig(
                shape=(4, 4), pixel_size=3.324, voltage_in_kilovolts=121.3
            ),
            pose=pose,
            transfer_theory=cxs.ContrastTransferTheory(
                cxs.CTF(defocus_in_angstroms=1234.0),
                amplitude_contrast_ratio=0.1234,
                envelope=op.FourierGaussian(b_factor=12.34) if sets_envelope else None,
            ),
            metadata=metadata,
        )

    metadata = pd.DataFrame(
        data={
            "rlnMicrographName": list("dummy/micrograph.mrc" for _ in range(index.size)),
            "rlnCoordinateX": np.atleast_1d(np.full_like(index, 2, dtype=int)),
            "rlnCoordinateY": np.atleast_1d(np.full_like(index, 1, dtype=int)),
        },
    )
    rng_key = jr.key(0)
    new_parameters = make_params(rng_key, metadata.to_dict())
    if ndim == 0:
        new_parameters = jax.tree.map(
            lambda x: jnp.squeeze(x) if isinstance(x, jax.Array) else x, new_parameters
        )

    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=sets_envelope,
        loads_metadata=False,
        updates_optics_group=updates_optics_group,
    )
    # Set params
    parameter_file[index] = new_parameters
    # Load params that were just set
    loaded_parameters = parameter_file[index]

    if updates_optics_group:
        assert compare_pytrees(
            eqx.tree_at(lambda x: x["metadata"], new_parameters, {}), loaded_parameters
        )
    else:
        assert compare_pytrees(new_parameters["pose"], loaded_parameters["pose"])
        np.testing.assert_allclose(
            new_parameters["transfer_theory"].ctf.defocus_in_angstroms,  # type: ignore
            loaded_parameters["transfer_theory"].ctf.defocus_in_angstroms,  # type: ignore
        )
        if sets_envelope:
            np.testing.assert_allclose(
                new_parameters["transfer_theory"].envelope.b_factor,  # type: ignore
                loaded_parameters["transfer_theory"].envelope.b_factor,  # type: ignore
            )
    # Make sure custom metadata was added
    particle_dataframe = parameter_file.starfile_data["particles"]
    assert set(metadata.columns).issubset(particle_dataframe.columns)
    # Make sure dataframes are the same
    metadata_extracted = particle_dataframe.loc[
        particle_dataframe.index[np.atleast_1d(index)],
        ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"],
    ]
    np.testing.assert_equal(metadata.to_numpy(), metadata_extracted.to_numpy())


def test_file_exists_error():
    # Create pytrees
    parameters = dict(
        instrument_config=cxs.InstrumentConfig(
            shape=(4, 4), pixel_size=1.1, voltage_in_kilovolts=300.0
        ),
        pose=cxs.EulerAnglePose(),
        transfer_theory=cxs.ContrastTransferTheory(ctf=cxs.CTF()),
    )
    # Add to dataset
    path_to_starfile = "tests/outputs/starfile_writing/test_particle_parameters.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exists_ok=True,
    )
    parameter_file.append(parameters)
    parameter_file.save(overwrite=True)

    # Test no exists_ok
    with pytest.raises(FileExistsError):
        _ = RelionParticleParameterFile(
            path_to_starfile=path_to_starfile,
            mode="w",
            exists_ok=False,
        )
    # Clean up
    shutil.rmtree(parameter_file.path_to_output.parent)


def test_file_not_found_error():
    dummy_path_to_starfile = "path/to/nonexistant/dir/nonexistant_file.star"

    # Test no exists_ok
    with pytest.raises(FileNotFoundError):
        _ = RelionParticleParameterFile(
            path_to_starfile=dummy_path_to_starfile,
            mode="r",
        )

    return


def test_set_wrong_parameters_error():
    # Wrong parameters
    wrong_pose = cxs.QuaternionPose()
    wrong_transfer_theory_1 = cxs.ContrastTransferTheory(ctf=cxs.NullCTF())
    wrong_transfer_theory_2 = cxs.ContrastTransferTheory(
        ctf=cxs.CTF(), envelope=op.ZeroMode()
    )
    # Right parameters
    right_pose = cxs.EulerAnglePose()
    right_transfer_theory = cxs.ContrastTransferTheory(ctf=cxs.CTF())
    instrument_config = cxs.InstrumentConfig(
        shape=(4, 4), pixel_size=1.1, voltage_in_kilovolts=300.0
    )
    # Create pytrees
    wrong_parameters_1 = dict(
        instrument_config=instrument_config,
        pose=right_pose,
        transfer_theory=wrong_transfer_theory_1,
    )
    wrong_parameters_2 = dict(
        instrument_config=instrument_config,
        pose=right_pose,
        transfer_theory=wrong_transfer_theory_2,
    )
    temp = dict(
        instrument_config=instrument_config,
        pose=right_pose,
        transfer_theory=right_transfer_theory,
    )
    wrong_parameters_3 = eqx.tree_at(lambda x: x["pose"], temp, wrong_pose)
    # Now the parameter dataset
    # Add to dataset
    path_to_starfile = "path/to/dummy/project/and/starfile.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exists_ok=True,
    )

    with pytest.raises(ValueError):
        parameter_file.append(wrong_parameters_1)

    with pytest.raises(ValueError):
        parameter_file.append(wrong_parameters_2)

    with pytest.raises(TypeError):
        parameter_file.append(wrong_parameters_3)


def test_bad_pytree_error():
    # Right parameters
    make_pose = eqx.filter_vmap(
        lambda x, y, phi, theta, psi: cxs.EulerAnglePose(x, y, phi, theta, psi)
    )
    pose = make_pose(
        jnp.atleast_1d(1.0),
        jnp.atleast_1d(-1.0),
        jnp.atleast_1d(1.0),
        jnp.atleast_1d(2.0),
        jnp.atleast_1d(3.0),
    )
    pose = eqx.tree_at(lambda x: x.offset_x_in_angstroms, pose, jnp.asarray((1.0, 2.0)))
    transfer_theory = cxs.ContrastTransferTheory(ctf=cxs.CTF())
    instrument_config = cxs.InstrumentConfig(
        shape=(4, 4), pixel_size=1.1, voltage_in_kilovolts=300.0
    )
    # Create pytrees
    parameters = dict(
        instrument_config=instrument_config,
        pose=pose,
        transfer_theory=transfer_theory,
    )
    # Now the parameter dataset
    # Add to dataset
    path_to_starfile = "path/to/dummy/project/and/starfile.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        exists_ok=True,
    )

    with pytest.raises(ValueError):
        parameter_file.append(parameters)


def test_write_image(
    sample_relion_project_path,
    sample_starfile_path,
    relion_parameters,
):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        exists_ok=True,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    dataset = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project=sample_relion_project_path,
        mode="w",
    )
    starfile_data = dataset.parameter_file.starfile_data
    assert starfile_data["particles"]["rlnImageName"].isna().all()

    shape = relion_parameters["instrument_config"].shape
    particle = dict(
        parameters=relion_parameters,
        images=jnp.zeros(shape, dtype=np.float32),
    )
    bad_shape_particle = dict(
        parameters=relion_parameters,
        images=jnp.zeros((shape[0], shape[1] + 1), dtype=np.float32),
    )
    bad_dim_particle = eqx.tree_at(
        lambda x: x["images"], bad_shape_particle, jnp.zeros(shape[0], dtype=np.float32)
    )

    with pytest.raises(ValueError):
        dataset[0] = bad_shape_particle

    with pytest.raises(TypeCheckError):
        dataset[0] = bad_dim_particle

    with pytest.raises(IOError):
        dataset[0] = particle

    dataset.mrcfile_settings = dict(prefix="f", overwrite=True)
    dataset[0] = particle

    starfile_data = dataset.parameter_file.starfile_data
    rln_image_name = starfile_data["particles"]["rlnImageName"][0]
    # Assert entry was written
    assert not pd.isna(rln_image_name)
    assert starfile_data["particles"]["rlnImageName"][1:].isna().all()
    # Assert file was written and delete it
    filename = str(rln_image_name).split("@")[1]
    path_to_filename = os.path.join(sample_relion_project_path, filename)
    assert os.path.exists(path_to_filename)
    os.remove(path_to_filename)
    assert not os.path.exists(path_to_filename)


def test_write_particle_batched_particle_parameters():
    @partial(eqx.filter_vmap, in_axes=(0), out_axes=eqx.if_array(0))
    def _make_particle_params(dummy_idx):
        instrument_config = cxs.InstrumentConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.CTF(), envelope=op.FourierGaussian()
        )
        return {
            "instrument_config": instrument_config,
            "pose": pose,
            "transfer_theory": transfer_theory,
            "metadata": {},
        }

    particle_params = _make_particle_params(jnp.array([0, 0, 0, 0, 0]))
    new_parameters_file = RelionParticleParameterFile(
        path_to_starfile="dummy.star",
        mode="w",
        exists_ok=True,
        updates_optics_group=True,
        loads_envelope=True,
    )

    new_parameters_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)
    # and try to save again
    with pytest.raises(FileExistsError):
        new_parameters_file.save(overwrite=False)

    parameter_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="r",
        loads_envelope=True,
        loads_metadata=False,
    )

    loaded_params = parameter_file[:]
    for key in particle_params:
        assert compare_pytrees(
            loaded_params[key], particle_params[key]
        ), f"Mismatch for {key}"
    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_starfile_different_envs():
    def _make_particle_params(envelope):
        instrument_config = cxs.InstrumentConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.CTF(),
            envelope=envelope,
        )
        return {
            "instrument_config": instrument_config,
            "pose": pose,
            "transfer_theory": transfer_theory,
            "metadata": {},
        }

    particle_params = _make_particle_params(op.FourierGaussian())
    new_parameters_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="w",
        exists_ok=True,
        updates_optics_group=True,
        loads_envelope=True,
    )

    particle_params = _make_particle_params(op.Constant(1.0))
    new_parameters_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="w",
        exists_ok=True,
        updates_optics_group=True,
        loads_envelope=True,
    )
    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)

    particle_params = _make_particle_params(None)
    new_parameters_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="w",
        exists_ok=True,
        updates_optics_group=True,
        loads_envelope=True,
    )
    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)

    with pytest.raises(ValueError):
        particle_params = _make_particle_params(op.ZeroMode(1.0))
        new_parameters_file = RelionParticleParameterFile(
            path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
            mode="w",
            exists_ok=True,
            updates_optics_group=True,
            loads_envelope=True,
        )
        new_parameters_file.append(particle_params)
        new_parameters_file.save(overwrite=True)

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_simulated_image_stack_from_starfile_jit(sample_starfile_path):
    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        return per_particle_args

    """Test writing a simulated image stack from a starfile."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    n_images = len(parameter_file)
    shape = parameter_file[0]["instrument_config"].shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    # Create a simulated image stack
    new_stack = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    simulate_particle_stack(
        new_stack,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        is_jittable=True,
        overwrite=True,
    )

    # try to overwrite
    simulate_particle_stack(
        new_stack,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        is_jittable=True,
        overwrite=True,
    )

    # Now trigger overwrite error
    with pytest.raises(FileExistsError):
        simulate_particle_stack(
            new_stack,
            compute_image_fn=_mock_compute_image,
            constant_args=(1.0, 2.0),
            per_particle_args=true_images,
            is_jittable=True,
            overwrite=False,
        )

    # load the simulated image stack
    particle_dataset = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="r",
    )

    images = particle_dataset[:]["images"]
    np.testing.assert_allclose(
        images,
        true_images.astype(jnp.float32),
    )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_simulated_image_stack_from_starfile_nojit(sample_starfile_path):
    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        c1, c2 = constant_args
        image = per_particle_args
        return image / np.linalg.norm(image)

    """Test writing a simulated image stack from a starfile."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    n_images = len(parameter_file)
    shape = parameter_file[0]["instrument_config"].shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    # Create a simulated image stack
    new_stack = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    simulate_particle_stack(
        new_stack,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        overwrite=True,
    )

    particle_dataset = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="r",
    )
    images = particle_dataset[:]["images"]
    np.testing.assert_allclose(
        images,
        true_images / np.linalg.norm(true_images, axis=(1, 2), keepdims=True),
    )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_single_image(sample_starfile_path):
    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        c1, c2 = constant_args
        p1, p2 = per_particle_args
        image = jnp.ones(
            particle_parameters["instrument_config"].shape, dtype=jnp.float32
        )
        return image / np.linalg.norm(image)

    selection_filter = {
        "rlnImageName": lambda x: np.where(x == "0000001@000000.mrcs", True, False)
    }
    """Test writing a simulated image stack from a starfile."""
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
        selection_filter=selection_filter,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    # Create a simulated image stack
    new_stack = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    n_images = 1
    simulate_particle_stack(
        new_stack,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=(3.0 * jnp.ones(n_images), 4.0 * jnp.ones(n_images)),
        overwrite=True,
        images_per_file=1,
    )

    particle_dataset = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="r",
    )
    images = particle_dataset[:]["images"]
    np.testing.assert_allclose(
        images,
        np.ones_like(images) / np.linalg.norm(np.ones_like(images)),
    )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_load_multiple_mrcs():
    @partial(eqx.filter_vmap, in_axes=(0), out_axes=eqx.if_array(0))
    def _make_particle_params(dummy_idx):
        instrument_config = cxs.InstrumentConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.CTF(), envelope=op.FourierGaussian()
        )
        return {
            "instrument_config": instrument_config,
            "pose": pose,
            "transfer_theory": transfer_theory,
            "metadata": {},
        }

    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        return per_particle_args

    particle_params = _make_particle_params(jnp.ones(10))
    new_parameters_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="w",
        exists_ok=True,
        updates_optics_group=True,
        loads_envelope=True,
    )
    new_parameters_file.append(particle_params)
    new_parameters_file.save(overwrite=True)

    parameter_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="r",
        loads_envelope=True,
        loads_metadata=False,
    )

    n_images = len(parameter_file)
    shape = parameter_file[0]["instrument_config"].shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    new_stack = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": True},
    )

    # Create a simulated image stack
    simulate_particle_stack(
        new_stack,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        overwrite=True,
        images_per_file=3,
        batch_size=1,
    )

    parameter_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="r",
        loads_envelope=True,
        loads_metadata=False,
    )

    particle_dataset = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="r",
    )

    n_tests = 10
    for i in range(n_tests):
        indices = np.random.choice(len(parameter_file), size=3, replace=False)

        images = particle_dataset[indices]["images"]
        np.testing.assert_allclose(
            images,
            true_images[indices],
        )

    shutil.rmtree("tests/outputs/starfile_writing/")
    return


def test_raise_errors_parameter_file(sample_starfile_path):
    from jaxtyping import TypeCheckError

    with pytest.raises((ValueError, TypeCheckError)):
        parameter_file = RelionParticleParameterFile(
            path_to_starfile=sample_starfile_path,
            mode="CRYOEM",
            loads_envelope=False,
            loads_metadata=False,
        )
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
    )

    assert parameter_file.mode == "r"

    # bad keys, bad values
    with pytest.raises((ValueError, TypeCheckError)):
        starfile_data = {"cryo": pd.DataFrame({}), "em": pd.DataFrame({})}
        parameter_file.starfile_data = starfile_data

    # good keys, bad values
    with pytest.raises((ValueError, TypeCheckError)):
        starfile_data = {"particles": 0, "optics": 0}
        parameter_file.starfile_data = starfile_data

    # now set to write mode and try to filter
    with pytest.raises(ValueError):
        parameter_file = RelionParticleParameterFile(
            path_to_starfile=sample_starfile_path,
            mode="w",
            exists_ok=True,
            loads_envelope=False,
            loads_metadata=False,
            selection_filter={"rlnAngleRot": lambda x: x < 1000.0},
        )


def test_raise_errors_stack_dataset(sample_starfile_path, sample_relion_project_path):
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=sample_starfile_path,
        mode="r",
        loads_envelope=False,
        loads_metadata=False,
    )
    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )

    starfile_data = parameter_file.starfile_data
    particle_data = starfile_data["particles"]
    # remove "rlnImageName" column
    particle_data = particle_data.drop(columns=["rlnImageName"])

    parameter_file.starfile_data = {
        "particles": particle_data,
        "optics": starfile_data["optics"],
    }

    with pytest.raises(IOError):
        particle_dataset = RelionParticleStackDataset(
            parameter_file,
            path_to_relion_project=sample_relion_project_path,
            mode="r",
        )

    # Now set to write mode

    parameter_file.path_to_starfile = (
        "tests/outputs/starfile_writing/test_particle_parameters.star"
    )
    particle_dataset = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": False},
    )

    parameters = parameter_file[0]
    image_shape = parameters["instrument_config"].shape

    particle_stack = {
        "parameters": parameters,
        "images": jnp.zeros(image_shape, dtype=np.float32),
    }

    with pytest.raises(ValueError):
        particle_dataset[np.array([0])] = particle_stack

    with pytest.raises(TypeError):
        particle_dataset[0] = "dummy"

    with pytest.raises(TypeError):
        particle_dataset.append("dummy")

    with pytest.raises(ValueError):
        particle_dataset.append({"parameters": None, "images": particle_stack["images"]})

    with pytest.raises(ValueError):
        particle_dataset.write_images(
            index_array=np.array([0, 1], dtype=int), images=np.zeros((100, 10, 10))
        )

    with pytest.raises(ValueError):
        particle_dataset.write_images(
            index_array=np.array([0, 1], dtype=int), images=np.zeros((10, *image_shape))
        )

    # and clean
    shutil.rmtree("tests/outputs/starfile_writing/")


def test_append_relion_stack_dataset():
    @partial(eqx.filter_vmap, in_axes=(0), out_axes=eqx.if_array(0))
    def _make_particle_params(dummy_idx):
        instrument_config = cxs.InstrumentConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.CTF(), envelope=op.FourierGaussian()
        )
        return {
            "instrument_config": instrument_config,
            "pose": pose,
            "transfer_theory": transfer_theory,
            "metadata": {},
        }

    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        return per_particle_args

    new_parameters_file = RelionParticleParameterFile(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        mode="w",
        exists_ok=True,
        updates_optics_group=True,
        loads_envelope=True,
    )

    new_stack = RelionParticleStackDataset(
        new_parameters_file,
        path_to_relion_project="tests/outputs/starfile_writing/",
        mode="w",
        mrcfile_settings={"overwrite": False},
    )

    n_images = 10
    particle_params = _make_particle_params(jnp.ones(n_images))
    shape = particle_params["instrument_config"].shape
    images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    new_stack.append(
        {
            "parameters": particle_params,
            "images": images,
        }
    )

    # clean up
    shutil.rmtree("tests/outputs/starfile_writing/")
    return
