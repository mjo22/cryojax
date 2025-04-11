"""Functions common to multiple scattering theories."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact


# Not currently public API
def apply_interaction_constant(
    integrated_potential: Inexact[Array, "y_dim x_dim"],
    wavelength_in_angstroms: Float[Array, ""] | float,
) -> Inexact[Array, "y_dim x_dim"]:
    """Given an integrated potential, convert units to the object
    phase shift distribution using the interaction
    constant. For example, the case of the projection approximation,
    compute the phase shift distribution.

    !!! info
        In the projection approximation in cryo-EM, the phase shifts in the
        exit plane are given by

        $$\\eta(x, y) = \\sigma \\int dz \\ V(x, y, z),$$

        where $\\sigma$ is typically referred to as the interaction
        constant. However, in `cryojax`, the potential is rescaled
        to units of inverse length squared as

        $$U(x, y, z) = \\frac{2 m e}{\\hbar^2} V(x, y, z).$$

        With this rescaling of the potential, the phase shifts are equal to

        $$\\eta(x, y) = \\frac{\\lambda}{4 \\pi} \\int dz \\ U(x, y, z).$$

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
    return integrated_potential * jnp.asarray(wavelength_in_angstroms) / (4 * jnp.pi)


# Not currently public API
def apply_amplitude_contrast_ratio(
    object_at_exit_plane: Float[Array, "y_dim x_dim"],
    amplitude_contrast_ratio: Float[Array, ""] | float,
) -> Float[Array, "y_dim x_dim"]:
    """Apply the amplitude contrast ratio to compute a complex object spectrum,
    given the object spectrum computed just with a weak-phase calculation.

    !!! info
        Given phase shift spectrum $\\eta(x, y)$ the complex object spectrum $O(x, y)$
        for amplitude contrast ratio $\\alpha$ is

        $$O(x, y) = -\\alpha \\ \\eta(x, y) + i \\sqrt{1 - \\alpha^2} \\ \\eta(x, y)$$
    """
    ac = amplitude_contrast_ratio
    return (jnp.sqrt(1.0 - ac**2) + 1.0j * ac) * object_at_exit_plane
