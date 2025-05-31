"""
Check coverage with
pytest --cov-report term-missing:skip-covered --cov=src/cryojax/data/_relion tests/test_relion_data.py
"""  # noqa

import os
import shutil
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
    RelionParticleParameters,
    RelionParticleStack,
    RelionParticleStackDataset,
)
from cryojax.data._relion._starfile_dataset import (
    _default_make_config_fn,
    _format_number_for_filename,
    _load_image_stack_from_mrc,
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
    return RelionParticleParameters(instrument_config, pose, transfer_theory)


#
# Tests for starfile loading
#
class TestErrorRaisingForLoading:
    def test_load_with_badparticle_name(self, parameter_file, sample_relion_project_path):
        with pytest.raises(IOError):
            metadata = parameter_file[0].metadata
            particle_dataframe_at_index = pd.DataFrame.from_dict(metadata)
            particle_dataframe_at_index["rlnImageName"] = 0.0

            _load_image_stack_from_mrc(
                0,
                particle_dataframe_at_index,
                sample_relion_project_path,
            )

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


def test_default_make_config_fn():
    """Test the default make_config_fn function."""
    # Test with a valid input
    config = _default_make_config_fn(
        shape=(128, 128),
        pixel_size=jnp.asarray(1.5),
        voltage_in_kilovolts=jnp.asarray(300.0),
        padded_shape=(140, 140),
        pad_mode="constant",
    )

    ref_config = cxs.InstrumentConfig(
        shape=(128, 128),
        pixel_size=1.5,
        voltage_in_kilovolts=300.0,
        padded_shape=(140, 140),
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
    parameter = parameter_file[0]
    assert parameter.transfer_theory.envelope is not None

    parameters = parameter_file[:]
    assert parameters.transfer_theory.envelope is not None

    envelope = parameters.transfer_theory.envelope
    # check that envelope params match
    for i in range(len(parameter_file)):
        # check b-factors
        np.testing.assert_allclose(
            envelope.b_factor[i],  # type: ignore
            parameters.metadata["rlnCtfBfactor"][i],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            envelope.amplitude[i],  # type: ignore
            parameters.metadata["rlnCtfScalefactor"][i],
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

    parameter = parameter_file[0]
    assert parameter.transfer_theory.envelope is None

    parameters = parameter_file[:]
    assert parameters.transfer_theory.envelope is None

    transfer_theory = parameters.transfer_theory
    ctf = cast(cxs.AberratedAstigmaticCTF, transfer_theory.ctf)

    # check CTF parameters
    for i in range(len(parameter_file)):
        # defocus
        np.testing.assert_allclose(
            ctf.defocus_in_angstroms[i],
            compute_defocus(
                parameters.metadata["rlnDefocusU"][i],
                parameters.metadata["rlnDefocusV"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism
        np.testing.assert_allclose(
            ctf.astigmatism_in_angstroms[i],
            compute_astigmatism(
                parameters.metadata["rlnDefocusU"][i],
                parameters.metadata["rlnDefocusV"][i],
            ),
            rtol=1e-5,
        )

        # astigmatism_angle
        np.testing.assert_allclose(
            ctf.astigmatism_angle[i],
            parameters.metadata["rlnDefocusAngle"][i],
            rtol=1e-5,
        )

        # phase shift
        np.testing.assert_allclose(
            transfer_theory.phase_shift[i],
            parameters.metadata["rlnPhaseShift"][i],
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
    pose = parameters.pose

    # check pose parameters
    for i in range(len(parameter_file)):
        # offset x
        np.testing.assert_allclose(
            pose.offset_x_in_angstroms[i],
            -parameters.metadata["rlnOriginXAngst"][i],  # conventions!
            rtol=1e-5,
        )

        # offset y
        np.testing.assert_allclose(
            pose.offset_y_in_angstroms[i],
            -parameters.metadata["rlnOriginYAngst"][i],  # conventions!
            rtol=1e-5,
        )

        # phi angle - AngleRot
        np.testing.assert_allclose(
            pose.phi_angle[i],
            -parameters.metadata["rlnAngleRot"][i],
            rtol=1e-5,
        )

        # theta angle - AngleTilt
        np.testing.assert_allclose(
            pose.theta_angle[i],
            -parameters.metadata["rlnAngleTilt"][i],
            rtol=1e-5,
        )

        # psi angle - AnglePsi
        np.testing.assert_allclose(
            pose.psi_angle[i],
            -parameters.metadata["rlnAnglePsi"][i],
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
    assert parameter_file[0].metadata == {}
    assert parameter_file[:].metadata == {}
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
    instrument_config = parameters.instrument_config
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
    instrument_config = parameters.instrument_config
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
    instrument_config = particle_stack.parameters.instrument_config
    assert particle_stack.images.shape == (
        len(parameter_file),
        *instrument_config.shape,
    )

    particle_stack = dataset[0]
    instrument_config = particle_stack.parameters.instrument_config
    assert particle_stack.images.shape == instrument_config.shape

    particle_stack = dataset[0:2]
    instrument_config = particle_stack.parameters.instrument_config
    assert particle_stack.images.shape == (2, *instrument_config.shape)

    assert len(dataset) == len(parameter_file)

    return


#
# Tests for starfile writing
#
def test_format_filename_for_mrcs():
    formated_number = _format_number_for_filename(10, n_characters=5)

    assert formated_number == "00010"

    formated_number = _format_number_for_filename(0, n_characters=5)
    assert formated_number == "00000"


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
    def make_particle_params(dummy_idx):
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
        return RelionParticleParameters(
            instrument_config, pose, transfer_theory, metadata={}
        )

    # Make particle parameters
    particle_params = make_particle_params(jnp.atleast_1d(index))
    if ndim == 0:
        particle_params = jax.tree.map(
            lambda x: jnp.squeeze(x) if isinstance(x, jax.Array) else x, particle_params
        )
    # Add to dataset
    path_to_starfile = "tests/outputs/starfile_writing/test_particle_parameters.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        overwrite=True,
        loads_envelope=loads_envelope,
    )
    parameter_file.append(particle_params)

    assert compare_pytrees(parameter_file[index], particle_params)


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

    def make_params(rng_key):
        rng_keys = jr.split(rng_key, n_particles)
        make_pose = eqx.filter_vmap(
            lambda rng_key: cxs.EulerAnglePose.from_rotation(SO3.sample_uniform(rng_key))
        )
        pose = make_pose(rng_keys)
        return RelionParticleParameters(
            instrument_config=cxs.InstrumentConfig(
                shape=(4, 4), pixel_size=3.324, voltage_in_kilovolts=121.3
            ),
            pose=pose,
            transfer_theory=cxs.ContrastTransferTheory(
                cxs.CTF(defocus_in_angstroms=1234.0),
                amplitude_contrast_ratio=0.1234,
                envelope=op.FourierGaussian(b_factor=12.34) if sets_envelope else None,
            ),
        )

    rng_key = jr.key(0)
    new_parameters = make_params(rng_key)
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
        assert compare_pytrees(new_parameters, loaded_parameters)
    else:
        assert compare_pytrees(new_parameters.pose, loaded_parameters.pose)
        np.testing.assert_allclose(
            new_parameters.transfer_theory.ctf.defocus_in_angstroms,  # type: ignore
            loaded_parameters.transfer_theory.ctf.defocus_in_angstroms,  # type: ignore
        )
        if sets_envelope:
            np.testing.assert_allclose(
                new_parameters.transfer_theory.envelope.b_factor,  # type: ignore
                loaded_parameters.transfer_theory.envelope.b_factor,  # type: ignore
            )


def test_file_exists_error():
    # Create pytrees
    parameters = RelionParticleParameters(
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
        overwrite=True,
    )
    parameter_file.append(parameters)
    parameter_file.save(overwrite=True)

    # Test no overwrite
    with pytest.raises(FileExistsError):
        _ = RelionParticleParameterFile(
            path_to_starfile=path_to_starfile,
            mode="w",
            overwrite=False,
        )
    # Clean up
    shutil.rmtree(parameter_file.path_to_output.parent)


def test_file_not_found_error():
    dummy_path_to_starfile = "path/to/nonexistant/dir/nonexistant_file.star"

    # Test no overwrite
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
    wrong_parameters_1 = RelionParticleParameters(
        instrument_config=instrument_config,
        pose=right_pose,
        transfer_theory=wrong_transfer_theory_1,
    )
    wrong_parameters_2 = RelionParticleParameters(
        instrument_config=instrument_config,
        pose=right_pose,
        transfer_theory=wrong_transfer_theory_2,
    )
    temp = RelionParticleParameters(
        instrument_config=instrument_config,
        pose=right_pose,
        transfer_theory=right_transfer_theory,
    )
    wrong_parameters_3 = eqx.tree_at(lambda x: x.pose, temp, wrong_pose)
    # Now the parameter dataset
    # Add to dataset
    path_to_starfile = "path/to/dummy/project/and/starfile.star"
    parameter_file = RelionParticleParameterFile(
        path_to_starfile=path_to_starfile,
        mode="w",
        overwrite=True,
    )

    with pytest.raises(ValueError):
        parameter_file.append(wrong_parameters_1)

    with pytest.raises(ValueError):
        parameter_file.append(wrong_parameters_2)

    with pytest.raises(ValueError):
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
    parameters = RelionParticleParameters(
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
        overwrite=True,
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
        overwrite=True,
    )

    with pytest.raises(IOError):
        dataset = RelionParticleStackDataset(
            parameter_file,
            path_to_relion_project=sample_relion_project_path,
            mode="w",
        )

    dataset = RelionParticleStackDataset(
        parameter_file,
        path_to_relion_project=sample_relion_project_path,
        mode="w",
        overwrite=True,
    )
    starfile_data = dataset.parameter_file.starfile_data
    assert starfile_data["particles"]["rlnImageName"].isna().all()

    shape = relion_parameters.instrument_config.shape
    particle = RelionParticleStack(
        parameters=relion_parameters,
        images=jnp.zeros(shape, dtype=np.float32),
    )
    bad_shape_particle = RelionParticleStack(
        parameters=relion_parameters,
        images=jnp.zeros((shape[0], shape[1] + 1), dtype=np.float32),
    )
    bad_dim_particle = eqx.tree_at(
        lambda x: x.images, bad_shape_particle, jnp.zeros(shape[0], dtype=np.float32)
    )

    with pytest.raises(ValueError):
        dataset[0] = bad_shape_particle

    with pytest.raises(TypeCheckError):
        dataset[0] = bad_dim_particle

    with pytest.raises(IOError):
        dataset[0] = particle

    dataset.filename_settings = dict(prefix="f", overwrite=True)
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


# def test_write_particle_batched_particle_parameters():
#     @partial(eqx.filter_vmap, in_axes=(0), out_axes=eqx.if_array(0))
#     def _make_particle_params(dummy_idx):
#         instrument_config = cxs.InstrumentConfig(
#             shape=(4, 4),
#             pixel_size=1.5,
#             voltage_in_kilovolts=300.0,
#         )

#         pose = cxs.EulerAnglePose()
#         transfer_theory = cxs.ContrastTransferTheory(
#             ctf=cxs.CTF(), envelope=op.FourierGaussian()
#         )
#         return RelionParticleParameters(
#             instrument_config, pose, transfer_theory, metadata={}
#         )

#     particle_params = _make_particle_params(jnp.array([0, 0, 0, 0, 0]))

#     write_starfile_with_particle_parameters(
#         particle_parameters=particle_params,
#         filename="tests/outputs/starfile_writing/test_particle_parameters.star",
#         mrc_batch_size=2,
#         overwrite=True,
#     )

#     parameter_file = RelionParticleParameterFile(
#         path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
#         path_to_relion_project="tests/outputs/starfile_writing/",
#         loads_envelope=True,
#         loads_metadata=False,
#     )

#     assert compare_pytrees(parameter_file[:], particle_params)
#     # Clean up
#     shutil.rmtree("tests/outputs/starfile_writing/")

#     return


# def test_write_starfile_different_envs():
#     def _make_particle_params(envelope):
#         instrument_config = cxs.InstrumentConfig(
#             shape=(4, 4),
#             pixel_size=1.5,
#             voltage_in_kilovolts=300.0,
#         )

#         pose = cxs.EulerAnglePose()
#         transfer_theory = cxs.ContrastTransferTheory(
#             ctf=cxs.CTF(),
#             envelope=envelope,
#         )
#         return RelionParticleParameters(
#             instrument_config, pose, transfer_theory, metadata={}
#         )

#     particle_params = _make_particle_params(op.FourierGaussian())
#     write_starfile_with_particle_parameters(
#         particle_parameters=particle_params,
#         filename="tests/outputs/starfile_writing/test_particle_parameters.star",
#         mrc_batch_size=None,
#         overwrite=True,
#     )

#     particle_params = _make_particle_params(Constant(1.0))
#     write_starfile_with_particle_parameters(
#         particle_parameters=particle_params,
#         filename="tests/outputs/starfile_writing/test_particle_parameters.star",
#         mrc_batch_size=None,
#         overwrite=True,
#     )

#     particle_params = _make_particle_params(None)
#     write_starfile_with_particle_parameters(
#         particle_parameters=particle_params,
#         filename="tests/outputs/starfile_writing/test_particle_parameters.star",
#         mrc_batch_size=None,
#         overwrite=True,
#     )

#     with pytest.raises(NotImplementedError):
#         particle_params = _make_particle_params(ZeroMode(1.0))
#         write_starfile_with_particle_parameters(
#             particle_parameters=particle_params,
#             filename="tests/outputs/starfile_writing/test_particle_parameters.star",
#             mrc_batch_size=None,
#             overwrite=True,
#         )

#     # Clean up
#     shutil.rmtree("tests/outputs/starfile_writing/")

#     return


# def test_write_simulated_image_stack_from_starfile_jit(sample_starfile_path):
#     def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
#         # Mock the image computation
#         return per_particle_args

#     """Test writing a simulated image stack from a starfile."""
#     parameter_file = RelionParticleParameterFile(
#         path_to_starfile=sample_starfile_path,
#         path_to_relion_project="tests/outputs/starfile_writing/",
#         loads_envelope=False,
#         loads_metadata=False,
#     )

#     n_images = len(parameter_file)
#     shape = parameter_file[0].instrument_config.shape
#     true_images = jax.random.normal(
#         jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
#     )
#     # Create a simulated image stack
#     write_simulated_image_stack_from_starfile(
#         parameter_file=parameter_file,
#         compute_image_fn=_mock_compute_image,
#         constant_args=(1.0, 2.0),
#         per_particle_args=true_images,
#         is_jittable=True,
#         overwrite=True,
#     )

#     # try to overwrite
#     write_simulated_image_stack_from_starfile(
#         parameter_file=parameter_file,
#         compute_image_fn=_mock_compute_image,
#         constant_args=(1.0, 2.0),
#         per_particle_args=true_images,
#         is_jittable=True,
#         overwrite=True,
#     )

#     # Now trigger overwrite error
#     with pytest.raises(FileExistsError):
#         write_simulated_image_stack_from_starfile(
#             parameter_file=parameter_file,
#             compute_image_fn=_mock_compute_image,
#             constant_args=(1.0, 2.0),
#             per_particle_args=true_images,
#             is_jittable=True,
#             overwrite=False,
#         )

#     # load the simulated image stack
#     particle_dataset = RelionParticleStackDataset(parameter_file)
#     images = particle_dataset[:].images
#     np.testing.assert_allclose(
#         images,
#         true_images,
#     )

#     # Clean up
#     shutil.rmtree("tests/outputs/starfile_writing/")

#     return


# def test_write_simulated_image_stack_from_starfile_nojit(sample_starfile_path):
#     def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
#         # Mock the image computation
#         c1, c2 = constant_args
#         image = per_particle_args
#         return image / np.linalg.norm(image)

#     """Test writing a simulated image stack from a starfile."""
#     parameter_file = RelionParticleParameterFile(
#         path_to_starfile=sample_starfile_path,
#         path_to_relion_project="tests/outputs/starfile_writing/",
#         loads_envelope=False,
#         loads_metadata=False,
#     )

#     n_images = len(parameter_file)
#     shape = parameter_file[0].instrument_config.shape
#     true_images = jax.random.normal(
#         jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
#     )

#     # check jit fails
#     with pytest.raises(RuntimeError):
#         write_simulated_image_stack_from_starfile(
#             parameter_file=parameter_file,
#             compute_image_fn=_mock_compute_image,
#             constant_args=(1.0, 2.0),
#             per_particle_args=true_images,
#             is_jittable=True,
#             overwrite=True,
#         )

#     # check that non jit mode works
#     write_simulated_image_stack_from_starfile(
#         parameter_file=parameter_file,
#         compute_image_fn=_mock_compute_image,
#         constant_args=(1.0, 2.0),
#         per_particle_args=true_images,
#         is_jittable=False,
#         overwrite=True,
#     )

#     particle_dataset = RelionParticleStackDataset(parameter_file)
#     images = particle_dataset[:].images
#     np.testing.assert_allclose(
#         images,
#         true_images / np.linalg.norm(true_images, axis=(1, 2), keepdims=True),
#     )

#     # Clean up
#     shutil.rmtree("tests/outputs/starfile_writing/")

#     return


# def test_write_single_image(sample_starfile_path):
#     def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
#         # Mock the image computation
#         c1, c2 = constant_args
#         p1, p2 = per_particle_args
#         image = jnp.ones(particle_parameters.instrument_config.shape, dtype=jnp.float32)
#         return image / np.linalg.norm(image)

#     """Test writing a simulated image stack from a starfile."""
#     parameter_file = RelionParticleParameterFile(
#         path_to_starfile=sample_starfile_path,
#         path_to_relion_project="tests/outputs/starfile_writing/",
#         loads_envelope=False,
#         loads_metadata=False,
#     )

#     write_starfile_with_particle_parameters(
#         particle_parameters=parameter_file[0],
#         filename="tests/outputs/starfile_writing/test_particle_parameters.star",
#         mrc_batch_size=None,
#         overwrite=True,
#     )

#     parameter_file = RelionParticleParameterFile(
#         path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
#         path_to_relion_project="tests/outputs/starfile_writing/",
#         loads_envelope=False,
#         loads_metadata=False,
#     )

#     n_images = 1

#     # check jit fails
#     with pytest.raises(RuntimeError):
#         write_simulated_image_stack_from_starfile(
#             parameter_file=parameter_file,
#             compute_image_fn=_mock_compute_image,
#             constant_args=(1.0, 2.0),
#             per_particle_args=(3.0 * jnp.ones(n_images), 4.0 * jnp.ones(n_images)),
#             is_jittable=True,
#             overwrite=True,
#         )

#     # check that non jit mode works
#     write_simulated_image_stack_from_starfile(
#         parameter_file=parameter_file,
#         compute_image_fn=_mock_compute_image,
#         constant_args=(1.0, 2.0),
#         per_particle_args=(3.0 * jnp.ones(n_images), 4.0 * jnp.ones(n_images)),
#         is_jittable=False,
#         overwrite=True,
#     )

#     particle_dataset = RelionParticleStackDataset(parameter_file)
#     images = particle_dataset[:].images
#     np.testing.assert_allclose(
#         images,
#         np.ones_like(images) / np.linalg.norm(np.ones_like(images)),
#     )

#     # Clean up
#     shutil.rmtree("tests/outputs/starfile_writing/")

#     return


# def test_load_multiple_mrcs():
#     @partial(eqx.filter_vmap, in_axes=(0), out_axes=eqx.if_array(0))
#     def _make_particle_params(dummy_idx):
#         instrument_config = cxs.InstrumentConfig(
#             shape=(4, 4),
#             pixel_size=1.5,
#             voltage_in_kilovolts=300.0,
#         )

#         pose = cxs.EulerAnglePose()
#         transfer_theory = cxs.ContrastTransferTheory(
#             ctf=cxs.CTF(), envelope=op.FourierGaussian()
#         )
#         return RelionParticleParameters(
#             instrument_config, pose, transfer_theory, metadata={}
#         )

#     def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
#         # Mock the image computation
#         return per_particle_args

#     particle_params = _make_particle_params(jnp.ones(10))

#     write_starfile_with_particle_parameters(
#         particle_parameters=particle_params,
#         filename="tests/outputs/starfile_writing/test_particle_parameters.star",
#         mrc_batch_size=3,
#         overwrite=True,
#     )

#     parameter_file = RelionParticleParameterFile(
#         path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
#         path_to_relion_project="tests/outputs/starfile_writing/",
#         loads_envelope=True,
#         loads_metadata=False,
#     )

#     n_images = len(parameter_file)
#     shape = parameter_file[0].instrument_config.shape
#     true_images = jax.random.normal(
#         jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
#     )

#     # Create a simulated image stack
#     write_simulated_image_stack_from_starfile(
#         parameter_file=parameter_file,
#         compute_image_fn=_mock_compute_image,
#         constant_args=(1.0, 2.0),
#         per_particle_args=true_images,
#         is_jittable=True,
#         overwrite=True,
#     )

#     stack_dataset = RelionParticleStackDataset(parameter_file)

#     n_tests = 10
#     for i in range(n_tests):
#         indices = np.random.choice(len(parameter_file), size=3, replace=False)

#         images = stack_dataset[indices].images
#         np.testing.assert_allclose(
#             images,
#             true_images[indices],
#         )
#     return
