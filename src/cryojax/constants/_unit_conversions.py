"""Unit conversions."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def convert_keV_to_angstroms(
    energy_in_keV: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """Get the relativistic electron wavelength at a given accelerating voltage."""
    energy_in_eV = 1000.0 * energy_in_keV  # keV to eV
    return jnp.asarray(12.2643 / (energy_in_eV + 0.97845e-6 * energy_in_eV**2) ** 0.5)
