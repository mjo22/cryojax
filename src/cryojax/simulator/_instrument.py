"""
Abstraction of the electron microscope. This includes models
for the optics, electron dose, and detector.
"""

from typing import Optional

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Float

from .._errors import error_if_not_positive
from ..constants import convert_keV_to_angstroms


class Instrument(Module, strict=True):
    """An abstraction of an electron microscope.

    **Attributes:**

    - `voltage_in_kilovolts`: The accelerating voltage of the
                              instrument in kilovolts (kV).
    - `electron_per_angstrom_squared`: The integrated electron flux.
    """

    voltage_in_kilovolts: Float[Array, ""]
    electrons_per_angstrom_squared: Optional[Float[Array, ""]]

    def __init__(
        self,
        voltage_in_kilovolts: float | Float[Array, ""],
        electrons_per_angstroms_squared: Optional[Float[Array, ""]] = None,
    ):
        self.voltage_in_kilovolts = error_if_not_positive(
            jnp.asarray(voltage_in_kilovolts)
        )
        self.electrons_per_angstrom_squared = (
            None
            if electrons_per_angstroms_squared is None
            else error_if_not_positive(jnp.asarray(electrons_per_angstroms_squared))
        )

    @property
    def wavelength_in_angstroms(self) -> Float[Array, ""]:
        return convert_keV_to_angstroms(self.voltage_in_kilovolts)
