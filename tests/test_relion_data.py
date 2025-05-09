# src/cryojax/data/_relion/_starfile_dataset.py

### Finished ###
# src/cryojax/data/_relion/_starfile_dataset.py  53, 62, 67, 72, 77, 82, 87,


### ToDO ###
# src/cryojax/data/_relion/_starfile_dataset.py 63, 68, 73, 78, 83, 88, 322-333, 337, 340-350, 353, 357, 361, 366, 371, 376, 381, 386, 391, 479-482, 494, 508-511, 571-576, 580-587, 590, 696, 716-733, 739-750, 767, 780-789, 821-823 #noqa

# src/cryojax/data/_relion/_starfile_writing.py 25-30, 57-180, 317-376, 392-463, 479-511 #noqa

import pathlib

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    import cryojax.simulator as cxs
    from cryojax.data import (
        ParticleStack,
        RelionParticleParameterDataset,
        RelionParticleParameters,
        RelionParticleStackDataset,
    )
    from cryojax.data._relion._starfile_dataset import (
        _default_make_config_fn,
        _get_image_stack_from_mrc,
        _validate_starfile_data,
    )


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
        padded_shape=(140, 140),
        pad_mode="constant",
    )

    pose = cxs.EulerAnglePose()
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.CTF(),
    )
    return RelionParticleParameters(instrument_config, pose, transfer_theory)


class TestErrorRaising:
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

    def test_param_dataset_max_index_int(self, parameter_dataset):
        with pytest.raises(IndexError):
            parameter_dataset[len(parameter_dataset)]

    def test_stack_dataset_max_index_int(self, parameter_dataset):
        stack_dataset = RelionParticleStackDataset(parameter_dataset)
        with pytest.raises(IndexError):
            stack_dataset[len(stack_dataset)]

    def test_param_dataset_max_index_slice(self, parameter_dataset):
        with pytest.raises(IndexError):
            parameter_dataset[len(parameter_dataset) :]

    def test_stack_dataset_max_index_slice(self, parameter_dataset):
        stack_dataset = RelionParticleStackDataset(parameter_dataset)
        with pytest.raises(IndexError):
            stack_dataset[len(stack_dataset) :]

    def test_param_dataset_badintex_type(self, parameter_dataset):
        with pytest.raises(IndexError):
            parameter_dataset["wrong_index"]

    def test_stack_dataset_badindex_type(self, parameter_dataset):
        stack_dataset = RelionParticleStackDataset(parameter_dataset)
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
