from importlib.resources import files

import jax
import jax.numpy as jnp
from jaxtyping import install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    import cryojax.coordinates as cxc
    import cryojax.simulator as cxs
    from cryojax.image import compute_radially_averaged_powerspectrum, irfftn, rfftn

"""
Helper functions
"""


def compute_projection(potential, shape, voxel_size):
    real_voxel_grid = potential.as_real_voxel_grid(shape, voxel_size)
    voxel_potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        real_voxel_grid, voxel_size
    )

    @jax.jit
    def compute_potential(potential):
        potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
        instrument_config = cxs.InstrumentConfig(
            shape=potential.shape[0:2],
            pixel_size=potential.voxel_size,
            voltage_in_kilovolts=300.0,
        )
        fourier_integrated_potential = (
            potential_integrator.compute_fourier_integrated_potential(
                potential, instrument_config
            )
        )
        return irfftn(fourier_integrated_potential, s=instrument_config.shape)

    return compute_potential(voxel_potential)


def compute_power_spectrum(integrated_potential, shape, voxel_size):
    integrated_potential_unbiased = integrated_potential - jnp.mean(integrated_potential)
    freq_grid = cxc.make_frequency_grid(shape, voxel_size)
    radial_freq_grid = jnp.linalg.norm(freq_grid, axis=-1)
    fourier_image = rfftn(integrated_potential_unbiased)
    n_pixels = fourier_image.shape[0]
    spectrum, frequencies = compute_radially_averaged_powerspectrum(
        fourier_image,
        radial_freq_grid,
        voxel_size,
        maximum_frequency=1 / (2 * voxel_size),
    )
    return frequencies, spectrum / n_pixels


def analytical_parkhurst_gaussian_envelope(
    q, a1=0.199, s1=0.731, a2=0.801, s2=0.081, m=1 / 2.88
):
    """_summary_ #TODO: add docstring

    Args:
        q (_type_): _description_
        a1 (float, optional): _description_. Defaults to 0.199.
        s1 (float, optional): _description_. Defaults to 0.731.
        a2 (float, optional): _description_. Defaults to 0.801.
        s2 (float, optional): _description_. Defaults to 0.081.
        m (_type_, optional): _description_. Defaults to 1/2.88.

    Returns:
        _type_: _description_
    """
    return a1 * jnp.exp(-(q**2) / (2 * s1**2)) + a2 * jnp.exp(
        -((q - m) ** 2) / (2 * s2**2)
    )


def generate_ice_image(solvent, image_shape, voxel_size, randomseedkey):
    instrument_config = cxs.InstrumentConfig(
        shape=image_shape,
        pixel_size=voxel_size,
        voltage_in_kilovolts=300.0,
    )

    phaseshifts = solvent.sample_ice_spectrum(
        key=randomseedkey, instrument_config=instrument_config
    )
    return irfftn(phaseshifts, s=instrument_config.shape)


"""
1.  Assert that the power spectrum of a generated ice image has the
    shape (within 5% tolerance) as the Parkhurst gaussian envelope function
"""


def test_1d_powerspectrum():
    solvent = cxs.Parkhurst2024_ExperimentalIce()
    image_shape = (80, 80)
    voxel_size = 1
    randomseedkey = jax.random.PRNGKey(0)

    image = generate_ice_image(solvent, image_shape, voxel_size, randomseedkey)

    x1, y1 = compute_power_spectrum(image, shape=image_shape, voxel_size=voxel_size)
    y2 = analytical_parkhurst_gaussian_envelope(x1)

    assert jnp.allclose(
        y1 / jnp.max(y1), y2 / jnp.max(y2), rtol=0.05
    ), "Power spectrum envelope mismatch"


"""
2.  Test that the mean and std of the ice image are the expected value
    based on image_mv__relaxed_small_box_tip3.npy (1% tolerance)
"""


def test_ice_mean_and_variance():
    solvent = cxs.Parkhurst2024_ExperimentalIce()

    image_mv = jnp.load(
        files("cryojax.simulator.data") / "image_mv__relaxed_small_box_tip3p.npy"
    )

    image_shape = (80, 80)
    key = jax.random.PRNGKey(0)

    for i in range(image_mv.shape[0]):
        voxel_size, expected_mean, expected_var = image_mv[i]

        image = generate_ice_image(solvent, image_shape, voxel_size, key)

        mean = jnp.mean(image)
        std = jnp.std(image)

        assert jnp.isclose(
            mean, expected_mean, rtol=0.01
        ), f"Mean mismatch at voxel size {voxel_size}"
        assert jnp.isclose(
            std**2, expected_var, rtol=0.01
        ), f"Variance mismatch at voxel size {voxel_size}"


# # 4. test that the power spectrum is the same as a real ice image
# # use /tests/data/relaxed_small_box_tip3.pdb
# # TODO: they don't match soooo....
# def test_real_ice_image():
#     solvent = cxs.Parkhurst2024_ExperimentalIce2(N=9261 / 81**2)
#     image_shape = (81, 81)
#     voxel_size = 1
#     randomseedkey = jax.random.PRNGKey(0)
#     image = generate_ice_image(solvent, image_shape, voxel_size, randomseedkey)
#     x1, y1 = compute_power_spectrum(image, shape=image_shape, voxel_size=voxel_size)

#     fname = "water_81_coords.pdb"
#     atom_positions, atom_identities = read_atoms_from_pdb_or_cif(
#         fname, center=True, get_b_factors=False, atom_filter="not element H"
#     )
#     potential = cxs.PengAtomicPotential(atom_positions, atom_identities)
#     integrated_potential = compute_projection(potential, image_shape, voxel_size)
#     integrated_potential_cropped = integrated_potential[4:-4, 4:-4]
#     x2, y2 = compute_power_spectrum(
#         integrated_potential_cropped, integrated_potential_cropped.shape, voxel_size
#     )

#     assert jnp.allclose(y1, y2, atol=jnp.inf)


# # 3. test the magnitude of the power spectrum?
# # TODO: finish this test -- not sure what expected magnitude is
# def test_power_spectrum_magnitude():
#     solvent = cxs.Parkhurst2024_ExperimentalIce2(N = 9261/81**2)
#     image_shape = (81, 81)
#     voxel_size = 1
#     randomseedkey = jax.random.PRNGKey(0)

#     image = generate_ice_image(solvent, image_shape, voxel_size, randomseedkey)

#     x1, y1 = compute_power_spectrum(image, shape=image_shape, voxel_size=voxel_size)
