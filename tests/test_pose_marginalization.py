from glob import glob

import jax.numpy as jnp
import numpy as np

import cryojax.simulator as cxs
from cryojax.constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from cryojax.data._pose_quadrature.lebedev import build_lebedev_quadrature
from cryojax.inference import distributions as dist
from cryojax.io import read_atoms_from_pdb
from cryojax.simulator import (
    GaussianMixtureProjection,
    PengAtomicPotential,
)


def test_build_lebeev_grid():
    """https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/lebedev_003.txt"""
    fname = "src/cryojax/data/_pose_quadrature/lebedev_003.txt"
    euler_angles, weights_repeated_for_psis = build_lebedev_quadrature(fname)
    assert np.isclose(weights_repeated_for_psis.sum(), 1.0), "Weights should sum to 1"
    assert euler_angles.shape == (len(weights_repeated_for_psis), 3)


def test_benchmark(sample_pdb_path):
    atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
        sample_pdb_path,
        center=True,
        selection_string="not element H",
        loads_b_factors=True,
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )

    potential = PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
        b_factors=b_factors,
    )

    pose = cxs.EulerAnglePose(
        offset_x_in_angstroms=5.0,
        offset_y_in_angstroms=-3.0,
        phi_angle=20.0,
        theta_angle=80.0,
        psi_angle=-5.0,
    )

    structural_ensemble = cxs.SingleStructureEnsemble(potential, pose)

    pixel_size = jnp.array(3.0, dtype=jnp.float32)
    potential_integrator = GaussianMixtureProjection(upsampling_factor=2)

    # ... next, the contrast transfer theory
    ctf = cxs.CTF(
        defocus_in_angstroms=10000.0,
        astigmatism_in_angstroms=-100.0,
        astigmatism_angle=10.0,
    )

    transfer_theory = cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=0.1)

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble, potential_integrator, transfer_theory, solvent=None
    )

    instrument_config = cxs.InstrumentConfig(
        shape=(80, 80),
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )
    image_model = cxs.ContrastImageModel(instrument_config, scattering_theory)

    for lebedev_quadrature_fname in glob(
        "src/cryojax/data/_pose_quadrature/lebedev_0??.txt"
    ):
        # print(f"Testing Lebedev quadrature file: {lebedev_quadrature_fname}")
        marginalization_euler_angles_deg, lebedev_weights = build_lebedev_quadrature(
            lebedev_quadrature_fname
        )
        distribution = dist.IndependentGaussianPoseMarginalizedOut(
            image_model,
            signal_scale_factor=1.0,
            variance=1.0,
            normalizes_signal=True,
            marginalization_euler_angles_deg=marginalization_euler_angles_deg,
            lebedev_weights=lebedev_weights,
        )
        log_marginal_liklihood = distribution.log_likelihood(
            observed=distribution.compute_signal(),
        )

        assert log_marginal_liklihood.ndim == 0, "Expected a scalar"
        assert jnp.issubdtype(
            log_marginal_liklihood.dtype, jnp.floating
        ), "Expected a float dtype"
