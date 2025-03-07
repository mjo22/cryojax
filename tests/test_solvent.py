# import jax
# import jax.numpy as jnp

# import cryojax.coordinates as cjc
# import cryojax.simulator as cxs
# from cryojax.image import compute_radially_averaged_powerspectrum, irfftn, rfftn


# # 1. test that within 1000 trials you get close to envelope function


# def test_1d_powerspectrum():
#     def compute_power_spectrum(integrated_potential, shape, voxel_size):
#         integrated_potential_unbiased = integrated_potential - jnp.mean(
#             integrated_potential
#         )
#         freq_grid = cjc.make_frequency_grid(shape, voxel_size)
#         radial_freq_grid = jnp.linalg.norm(freq_grid, axis=-1)
#         fourier_image = rfftn(integrated_potential_unbiased)
#         n_pixels = fourier_image.shape[0]
#         spectrum, frequencies = compute_radially_averaged_powerspectrum(
#             fourier_image,
#             radial_freq_grid,
#             voxel_size,
#             maximum_frequency=1 / (2 * voxel_size),
#         )
#         return frequencies, spectrum / n_pixels

#     # SIMULATION
#     instrument_config = cxs.InstrumentConfig(
#         shape=(256, 256),
#         pixel_size=0.4,
#         voltage_in_kilovolts=300.0,
#     )

#     solvent = cxs.GaussianIce(variance_function=cxs.Parkhurst2024_Gaussian())

#     # Number of simulations
#     num_simulations = 1000
#     image_shape = (256, 256)
#     voxel_size = 1
#     power_spectrum_sum = 0

#     # Run simulations
#     for i in range(num_simulations):
#         randomseedkey = jax.random.split(jax.random.PRNGKey(0), num_simulations)[i]
#         phaseshifts = solvent.sample_ice_spectrum(
#             key=randomseedkey, instrument_config=instrument_config
#         )
#         image_jax = irfftn(phaseshifts, s=instrument_config.shape)
#         image_np = jnp.array(image_jax)
#         x2, y2 = compute_power_spectrum(
#             image_np, shape=image_shape, voxel_size=voxel_size
#         )
#         power_spectrum_sum += y2

#     # Compute the average power spectrum
#     average_power_spectrum = power_spectrum_sum / num_simulations

#     # ENVELOPE
#     def ParkhurstGaussian(q, a1, s1, a2, s2, m):
#         return a1 * jnp.exp(-(q**2) / (2 * s1**2)) + a2 * jnp.exp(
#             -((q - m) ** 2) / (2 * s2**2)
#         )

#     a1 = 0.199
#     s1 = 0.731
#     a2 = 0.801
#     s2 = 0.081
#     m = 1 / 2.88
#     P_q = ParkhurstGaussian(x2, a1, s1, a2, s2, m)

#     assert jnp.allclose(average_power_spectrum, P_q, atol=jnp.inf)
