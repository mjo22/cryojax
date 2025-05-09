"""
Check coverage with
pytest --cov-report term-missing:skip-covered --cov=src/cryojax/data/_relion tests/test_relion_data.py
"""  # noqa

import os
import pathlib
import shutil
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import starfile
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    import cryojax.simulator as cxs
    from cryojax.data import (
        ParticleStack,
        RelionParticleParameterDataset,
        RelionParticleParameters,
        RelionParticleStackDataset,
        write_simulated_image_stack_from_starfile,
        write_starfile_with_particle_parameters,
    )
    from cryojax.data._relion._starfile_dataset import (
        _default_make_config_fn,
        _get_image_stack_from_mrc,
        _validate_starfile_data,
    )
    from cryojax.data._relion._starfile_writing import (
        _format_string_for_filename,
        _validate_particle_parameters_pytrees,
    )
    from cryojax.image.operators import Constant, FourierGaussian, ZeroMode


def compare_pytrees(pytree1, pytree2):
    arrays1, others1 = eqx.partition(pytree1, eqx.is_array)
    arrays2, others2 = eqx.partition(pytree2, eqx.is_array)

    jax.tree.map(lambda x, y: jnp.allclose(x, y), arrays1, arrays2)

    bool_arrays = all(
        jax.tree.leaves(jax.tree.map(lambda x, y: jnp.allclose(x, y), arrays1, arrays2))
    )
    bool_others = all(
        jax.tree.leaves(jax.tree.map(lambda x, y: x == y, others1, others2))
    )
    return bool_arrays and bool_others


@pytest.fixture
def parameter_dataset(sample_starfile_path, sample_path_to_relion_project):
    return RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
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


class TestErrorRaisingForLoading:
    def test_param_dataset_setitem(self, parameter_dataset, relion_parameters):
        with pytest.raises(NotImplementedError):
            parameter_dataset[0] = relion_parameters

    def test_stack_dataset_setitem(self, parameter_dataset, relion_parameters):
        stack_dataset = RelionParticleStackDataset(parameter_dataset)
        particle_stack = ParticleStack(
            relion_parameters, images=jnp.zeros(relion_parameters.instrument_config.shape)
        )
        with pytest.raises(NotImplementedError):
            stack_dataset[0] = particle_stack

    def test_load_with_badparticle_name(self, parameter_dataset):
        with pytest.raises(IOError):
            metadata = parameter_dataset[0].metadata
            particle_dataframe_at_index = pd.DataFrame.from_dict(metadata)
            particle_dataframe_at_index["rlnImageName"] = 0.0

            _get_image_stack_from_mrc(
                0,
                particle_dataframe_at_index,
                parameter_dataset.path_to_relion_project,
            )

    def test_with_bad_indices(self, parameter_dataset):
        stack_dataset = RelionParticleStackDataset(parameter_dataset)

        # overflow index
        with pytest.raises(IndexError):
            parameter_dataset[len(parameter_dataset)]

        with pytest.raises(IndexError):
            stack_dataset[len(stack_dataset)]

        # overflow slice
        with pytest.raises(IndexError):
            parameter_dataset[len(parameter_dataset) :]

        with pytest.raises(IndexError):
            stack_dataset[len(stack_dataset) :]

        # wrong index type
        with pytest.raises(IndexError):
            parameter_dataset["wrong_index"]

        with pytest.raises(IndexError):
            stack_dataset["wrong_index"]

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
    instrument_config = _default_make_config_fn(
        shape=(128, 128),
        pixel_size=jnp.asarray(1.5),
        voltage_in_kilovolts=jnp.asarray(300.0),
        padded_shape=(140, 140),
        pad_mode="constant",
    )

    ref_instrument_config = cxs.InstrumentConfig(
        shape=(128, 128),
        pixel_size=1.5,
        voltage_in_kilovolts=300.0,
        padded_shape=(140, 140),
        pad_mode="constant",
    )

    assert instrument_config.shape == ref_instrument_config.shape
    assert instrument_config.pixel_size == ref_instrument_config.pixel_size
    assert (
        instrument_config.voltage_in_kilovolts
        == ref_instrument_config.voltage_in_kilovolts
    )
    assert (
        instrument_config.electrons_per_angstrom_squared
        == ref_instrument_config.electrons_per_angstrom_squared
    )

    assert instrument_config.padded_shape == ref_instrument_config.padded_shape
    assert instrument_config.pad_mode == ref_instrument_config.pad_mode

    return


def test_load_starfile_envelope_params(
    sample_starfile_path, sample_path_to_relion_project
):
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=True,
        loads_metadata=True,
    )

    assert parameter_dataset.loads_envelope is True
    parameter = parameter_dataset[0]
    assert parameter.transfer_theory.envelope is not None

    parameters = parameter_dataset[:]
    assert parameters.transfer_theory.envelope is not None

    envelope = parameters.transfer_theory.envelope
    # check that envelope params match
    for i in range(len(parameter_dataset)):
        # check b-factors
        np.testing.assert_allclose(
            envelope.b_factor[i],
            parameters.metadata["rlnCtfBfactor"][i],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            envelope.amplitude[i],
            parameters.metadata["rlnCtfScalefactor"][i],
            rtol=1e-5,
        )
    return


def test_load_starfile_ctf_params(sample_starfile_path, sample_path_to_relion_project):
    def compute_defocus(defU, defV):
        return 0.5 * (defU + defV)

    def compute_astigmatism(defU, defV):
        return defU - defV

    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=False,
        loads_metadata=True,
    )

    assert parameter_dataset.loads_envelope is False

    parameter = parameter_dataset[0]
    assert parameter.transfer_theory.envelope is None

    parameters = parameter_dataset[:]
    assert parameters.transfer_theory.envelope is None

    transfer_theory = parameters.transfer_theory
    ctf = transfer_theory.ctf

    # check CTF parameters
    for i in range(len(parameter_dataset)):
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


def test_load_starfile_pose_params(sample_starfile_path, sample_path_to_relion_project):
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=False,
        loads_metadata=True,
    )

    parameters = parameter_dataset[:]
    pose = parameters.pose

    # check pose parameters
    for i in range(len(parameter_dataset)):
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

    return


def test_load_starfile_wo_metadata(sample_starfile_path, sample_path_to_relion_project):
    """Test loading a starfile without metadata."""
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=False,
        loads_metadata=False,
    )

    # check that metadata is empty dict
    assert parameter_dataset[0].metadata == {}
    assert parameter_dataset[:].metadata == {}
    assert parameter_dataset.loads_metadata is False

    return


def test_load_starfile_optics_group(sample_starfile_path, sample_path_to_relion_project):
    """Test loading a starfile with optics group."""
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=False,
        loads_metadata=True,
        broadcasts_optics_group=True,
    )

    parameters = parameter_dataset[:]
    instrument_config = parameters.instrument_config
    assert instrument_config.voltage_in_kilovolts.ndim > 0
    assert instrument_config.pixel_size.ndim > 0
    assert parameter_dataset.broadcasts_optics_group is True

    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=False,
        loads_metadata=True,
        broadcasts_optics_group=False,
    )
    parameters = parameter_dataset[:]
    instrument_config = parameters.instrument_config
    assert instrument_config.voltage_in_kilovolts.ndim == 0
    assert instrument_config.pixel_size.ndim == 0
    assert parameter_dataset.broadcasts_optics_group is False

    return


def test_load_starfile_misc(sample_starfile_path, sample_path_to_relion_project):
    """Test loading a starfile with miscellaneous parameters."""
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=False,
        loads_metadata=False,
        broadcasts_optics_group=False,
    )

    assert parameter_dataset.path_to_relion_project == pathlib.Path(
        sample_path_to_relion_project
    )

    # set to True to load metadata
    parameter_dataset.loads_metadata = True
    assert parameter_dataset.loads_metadata is True

    # set to True to load envelope
    parameter_dataset.loads_envelope = True
    assert parameter_dataset.loads_envelope is True

    # set to True to load optics group
    parameter_dataset.broadcasts_optics_group = True
    assert parameter_dataset.broadcasts_optics_group is True


def test_load_starfile_and_mrcs(sample_starfile_path, sample_path_to_relion_project):
    """Test loading a starfile with mrcs."""
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project=sample_path_to_relion_project,
        loads_envelope=False,
        loads_metadata=False,
        broadcasts_optics_group=False,
    )

    particle_stack_dataset = RelionParticleStackDataset(param_dataset=parameter_dataset)

    particle_stack = particle_stack_dataset[:]
    instrument_config = particle_stack.parameters.instrument_config
    assert particle_stack.images.shape == (
        len(parameter_dataset),
        *instrument_config.shape,
    )

    particle_stack = particle_stack_dataset[0]
    instrument_config = particle_stack.parameters.instrument_config
    assert particle_stack.images.shape == instrument_config.shape

    particle_stack = particle_stack_dataset[0:2]
    instrument_config = particle_stack.parameters.instrument_config
    assert particle_stack.images.shape == (2, *instrument_config.shape)

    assert len(particle_stack_dataset) == len(parameter_dataset)

    return


def test_default_starfile():
    path_to_starfile = "tests/outputs/starfile_writing/"
    os.makedirs(path_to_starfile, exist_ok=True)

    n_images = 1
    starfile_dict = dict()

    # Optics
    optics_df = pd.DataFrame()

    optics_df["rlnOpticsGroup"] = [1]
    optics_df["rlnVoltage"] = 300.0
    optics_df["rlnSphericalAberration"] = 2.7
    optics_df["rlnAmplitudeContrast"] = 0.1
    optics_df["rlnImagePixelSize"] = 3.0
    optics_df["rlnImageSize"] = 16

    # Particles
    particles_df = pd.DataFrame()

    # Misc
    particles_df["rlnCtfMaxResolution"] = np.zeros(n_images)
    particles_df["rlnCtfFigureOfMerit"] = np.zeros(n_images)
    particles_df["rlnClassNumber"] = np.ones(n_images)
    particles_df["rlnOpticsGroup"] = np.ones(n_images)

    # CTF
    particles_df["rlnDefocusU"] = 10000.0
    particles_df["rlnDefocusV"] = 9000.0
    particles_df["rlnDefocusAngle"] = 0.0
    particles_df["rlnPhaseShift"] = 0.0

    # we skip pose, as this one has default values

    # Image name
    particles_df["rlnImageName"] = "test.mrcs"

    starfile_dict["optics"] = optics_df
    starfile_dict["particles"] = particles_df
    starfile.write(starfile_dict, os.path.join(path_to_starfile, "test.star"))

    # now load the starfile
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=os.path.join(path_to_starfile, "test.star"),
        path_to_relion_project=path_to_starfile,
        loads_envelope=False,
        loads_metadata=False,
        broadcasts_optics_group=False,
    )

    assert all(
        jax.tree.leaves(
            jax.tree.map(lambda x: jnp.isclose(x, 0.0), parameter_dataset[:].pose)
        )
    )

    # clean up
    os.remove(os.path.join(path_to_starfile, "test.star"))


# Starfile writing
def test_format_filename_for_mrcs():
    formated_number = _format_string_for_filename(10, total_characters=5)

    assert formated_number == "00010"

    formated_number = _format_string_for_filename(0, total_characters=5)
    assert formated_number == "00000"


def test_write_particle_parameters():
    def _make_particle_params(dummy_idx):
        instrument_config = cxs.InstrumentConfig(
            shape=(4, 4),
            pixel_size=1.5,
            voltage_in_kilovolts=300.0,
        )

        pose = cxs.EulerAnglePose()
        transfer_theory = cxs.ContrastTransferTheory(
            ctf=cxs.CTF(),
            envelope=FourierGaussian(),
        )
        return RelionParticleParameters(
            instrument_config, pose, transfer_theory, metadata={}
        )

    particle_params = _make_particle_params(0)

    write_starfile_with_particle_parameters(
        particle_parameters=particle_params,
        filename="tests/outputs/starfile_writing/test_particle_parameters.star",
        mrc_batch_size=None,
        overwrite=True,
    )

    param_dataset = RelionParticleParameterDataset(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        path_to_relion_project="tests/outputs/starfile_writing/",
        loads_envelope=True,
        loads_metadata=False,
    )

    assert compare_pytrees(param_dataset[0], particle_params)

    with pytest.raises(FileExistsError):
        write_starfile_with_particle_parameters(
            particle_parameters=particle_params,
            filename="tests/outputs/starfile_writing/test_particle_parameters.star",
            mrc_batch_size=None,
            overwrite=False,
        )

    # write multiple parameters
    particle_params = eqx.filter_vmap(
        _make_particle_params, in_axes=(0), out_axes=eqx.if_array(0)
    )
    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


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
            ctf=cxs.CTF(), envelope=FourierGaussian()
        )
        return RelionParticleParameters(
            instrument_config, pose, transfer_theory, metadata={}
        )

    particle_params = _make_particle_params(jnp.array([0, 0, 0, 0, 0]))

    write_starfile_with_particle_parameters(
        particle_parameters=particle_params,
        filename="tests/outputs/starfile_writing/test_particle_parameters.star",
        mrc_batch_size=2,
        overwrite=True,
    )

    param_dataset = RelionParticleParameterDataset(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        path_to_relion_project="tests/outputs/starfile_writing/",
        loads_envelope=True,
        loads_metadata=False,
    )

    assert compare_pytrees(param_dataset[:], particle_params)
    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_wrong_particle_parameters_pytrees():
    pose_quat = cxs.QuaternionPose()
    pose_euler = cxs.EulerAnglePose()

    transfer_theory = cxs.ContrastTransferTheory(ctf=cxs.CTF())
    transfer_theory_null = cxs.ContrastTransferTheory(ctf=cxs.NullCTF())

    with pytest.raises(NotImplementedError):
        _validate_particle_parameters_pytrees(pose_quat, transfer_theory)

    with pytest.raises(NotImplementedError):
        _validate_particle_parameters_pytrees(pose_euler, transfer_theory_null)


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
        return RelionParticleParameters(
            instrument_config, pose, transfer_theory, metadata={}
        )

    particle_params = _make_particle_params(FourierGaussian())
    write_starfile_with_particle_parameters(
        particle_parameters=particle_params,
        filename="tests/outputs/starfile_writing/test_particle_parameters.star",
        mrc_batch_size=None,
        overwrite=True,
    )

    particle_params = _make_particle_params(Constant(1.0))
    write_starfile_with_particle_parameters(
        particle_parameters=particle_params,
        filename="tests/outputs/starfile_writing/test_particle_parameters.star",
        mrc_batch_size=None,
        overwrite=True,
    )

    particle_params = _make_particle_params(None)
    write_starfile_with_particle_parameters(
        particle_parameters=particle_params,
        filename="tests/outputs/starfile_writing/test_particle_parameters.star",
        mrc_batch_size=None,
        overwrite=True,
    )

    with pytest.raises(NotImplementedError):
        particle_params = _make_particle_params(ZeroMode(1.0))
        write_starfile_with_particle_parameters(
            particle_parameters=particle_params,
            filename="tests/outputs/starfile_writing/test_particle_parameters.star",
            mrc_batch_size=None,
            overwrite=True,
        )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return


def test_write_simulated_image_stack_from_starfile_jit(sample_starfile_path):
    def _mock_compute_image(particle_parameters, constant_args, per_particle_args):
        # Mock the image computation
        return per_particle_args

    """Test writing a simulated image stack from a starfile."""
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project="tests/outputs/starfile_writing/",
        loads_envelope=False,
        loads_metadata=False,
    )

    n_images = len(parameter_dataset)
    shape = parameter_dataset[0].instrument_config.shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )
    # Create a simulated image stack
    write_simulated_image_stack_from_starfile(
        param_dataset=parameter_dataset,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        is_jittable=True,
        overwrite=True,
    )

    # try to overwrite
    write_simulated_image_stack_from_starfile(
        param_dataset=parameter_dataset,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        is_jittable=True,
        overwrite=True,
    )

    # Now trigger overwrite error
    with pytest.raises(FileExistsError):
        write_simulated_image_stack_from_starfile(
            param_dataset=parameter_dataset,
            compute_image_fn=_mock_compute_image,
            constant_args=(1.0, 2.0),
            per_particle_args=true_images,
            is_jittable=True,
            overwrite=False,
        )

    # load the simulated image stack
    particle_dataset = RelionParticleStackDataset(parameter_dataset)
    images = particle_dataset[:].images
    np.testing.assert_allclose(
        images,
        true_images,
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
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project="tests/outputs/starfile_writing/",
        loads_envelope=False,
        loads_metadata=False,
    )

    n_images = len(parameter_dataset)
    shape = parameter_dataset[0].instrument_config.shape
    true_images = jax.random.normal(
        jax.random.key(0), shape=(n_images, *shape), dtype=jnp.float32
    )

    # check jit fails
    with pytest.raises(RuntimeError):
        write_simulated_image_stack_from_starfile(
            param_dataset=parameter_dataset,
            compute_image_fn=_mock_compute_image,
            constant_args=(1.0, 2.0),
            per_particle_args=true_images,
            is_jittable=True,
            overwrite=True,
        )

    # check that non jit mode works
    write_simulated_image_stack_from_starfile(
        param_dataset=parameter_dataset,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=true_images,
        is_jittable=False,
        overwrite=True,
    )

    particle_dataset = RelionParticleStackDataset(parameter_dataset)
    images = particle_dataset[:].images
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
        image = jnp.ones(particle_parameters.instrument_config.shape, dtype=jnp.float32)
        return image / np.linalg.norm(image)

    """Test writing a simulated image stack from a starfile."""
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=sample_starfile_path,
        path_to_relion_project="tests/outputs/starfile_writing/",
        loads_envelope=False,
        loads_metadata=False,
    )

    write_starfile_with_particle_parameters(
        particle_parameters=parameter_dataset[0],
        filename="tests/outputs/starfile_writing/test_particle_parameters.star",
        mrc_batch_size=None,
        overwrite=True,
    )

    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile="tests/outputs/starfile_writing/test_particle_parameters.star",
        path_to_relion_project="tests/outputs/starfile_writing/",
        loads_envelope=False,
        loads_metadata=False,
    )

    n_images = 1

    # check jit fails
    with pytest.raises(RuntimeError):
        write_simulated_image_stack_from_starfile(
            param_dataset=parameter_dataset,
            compute_image_fn=_mock_compute_image,
            constant_args=(1.0, 2.0),
            per_particle_args=(3.0 * jnp.ones(n_images), 4.0 * jnp.ones(n_images)),
            is_jittable=True,
            overwrite=True,
        )

    # check that non jit mode works
    write_simulated_image_stack_from_starfile(
        param_dataset=parameter_dataset,
        compute_image_fn=_mock_compute_image,
        constant_args=(1.0, 2.0),
        per_particle_args=(3.0 * jnp.ones(n_images), 4.0 * jnp.ones(n_images)),
        is_jittable=False,
        overwrite=True,
    )

    particle_dataset = RelionParticleStackDataset(parameter_dataset)
    images = particle_dataset[:].images
    np.testing.assert_allclose(
        images,
        np.ones_like(images) / np.linalg.norm(np.ones_like(images)),
    )

    # Clean up
    shutil.rmtree("tests/outputs/starfile_writing/")

    return
