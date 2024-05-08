"""Unit conversions."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def convert_keV_to_angstroms(
    energy_in_keV: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """Get the relativistic electron wavelength at a given accelerating voltage."""
    energy_in_eV = 1000.0 * energy_in_keV  # keV to eV
    return jnp.asarray(12.2643 / (energy_in_eV + 0.97845e-6 * energy_in_eV**2) ** 0.5)


def convert_wavelength_to_interaction_constant(
    wavelength_in_angstroms: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """Evaluate the interaction constant at a given electron wavelength.

    !!! info
        In `cryojax`, the interaction constant, usually denoted as
        $\\sigma$, is taken to be $\\lambda /  4 \\pi$, where $\\lambda$
        is the wavelength. This choice may seem odd, but this is because
        the potential is defined in units of inverse length squared as

        $$v(\\mathbf{x}) = \\frac{2 m e}{\\hbar^2} V(\\mathbf{x}).$$

        In the projection approximation in cryo-EM, the phase shifts in the
        exit plane are

        $$\\eta(x, y) = \\sigma \\int dz \\ V(x, y, z),$$

        With this rescaling of the potential, the phase shifts are equal to

        $$\\eta(x, y) = \\frac{\\lambda}{4 \\pi} \\int dz \\ v(x, y, z).$$

        For the sake of familiarity, we refer to this as a rescaling of the
        interaction constant.

        **References**:

        - For the definition of the rescaled potential, see
        Chapter 69, Page 2003, Equation 69.6 from *Hawkes, Peter W., and Erwin Kasper.
        Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic Press,
        2022.*
        - For the definition of the phase shifts in terms of the rescaled potential, see
        Chapter 69, Page 2012, Equation 69.34b from *Hawkes, Peter W., and Erwin Kasper.
        Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic Press,
        2022.*

    See the documentation on atom-based scattering potentials for more information.
    """  # noqa: E501
    return jnp.asarray(wavelength_in_angstroms) / (4 * jnp.pi)
