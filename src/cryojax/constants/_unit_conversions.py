"""Unit conversions."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def convert_keV_to_angstroms(
    energy_in_keV: Float[Array | np.ndarray, ""] | float,
) -> Float[Array, ""]:
    """Get the relativistic electron wavelength at a given accelerating voltage. For
    reference, see Equation 2.5 in Section 2.1 from *Spence, John CH. High-resolution
    electron microscopy. OUP Oxford, 2013.*.

    **Arguments:**

    - `energy_in_keV`:
        The energy in kiloelectron volts.

    **Returns:**

    The relativistically corrected electron wavelength in Angstroms corresponding to the
    energy `energy_in_keV`.
    """
    energy_in_eV = 1000.0 * energy_in_keV  # keV to eV
    return jnp.asarray(12.2639 / (energy_in_eV + 0.97845e-6 * energy_in_eV**2) ** 0.5)
