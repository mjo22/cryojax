import jax
import jax.numpy as jnp
import numpy as np

import cryojax.simulator as cs
from cryojax.image import irfftn, rfftn


def test_gaussian_limit():
    # Pick a large integrated electron flux to test
    electrons_per_angstrom_squared = 10000.0
    # Create ImageConfig
    config = cs.ImageConfig((25, 25), 1.0)
    N_pix = np.prod(config.padded_shape)
    electrons_per_pixel = electrons_per_angstrom_squared * config.pixel_size**2
    # Create squared wavefunction of just vacuum, i.e. 1 everywhere
    vacuum_squared_wavefunction = jnp.ones(config.shape, dtype=float)
    fourier_vacuum_squared_wavefunction = rfftn(vacuum_squared_wavefunction)
    # Instantiate the electron dose
    dose = cs.ElectronDose(electrons_per_angstrom_squared)
    # Create detector models
    key = jax.random.PRNGKey(1234)
    dqe = cs.NullDQE()
    gaussian_detector = cs.GaussianDetector(dqe)
    poisson_detector = cs.PoissonDetector(dqe)
    # Compute detector readout
    fourier_gaussian_detector_readout = gaussian_detector(
        fourier_vacuum_squared_wavefunction, dose, config, key
    )
    fourier_poisson_detector_readout = poisson_detector(
        fourier_vacuum_squared_wavefunction, dose, config, key
    )
    # Compare to see if the autocorrelation has converged
    np.testing.assert_allclose(
        irfftn(
            jnp.abs(fourier_gaussian_detector_readout) ** 2
            / (N_pix * electrons_per_pixel**2),
            s=config.padded_shape,
        ),
        irfftn(
            jnp.abs(fourier_poisson_detector_readout) ** 2
            / (N_pix * electrons_per_pixel**2),
            s=config.padded_shape,
        ),
        rtol=1e-2,
    )
