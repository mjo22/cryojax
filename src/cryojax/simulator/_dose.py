"""
Models the electron dose.
"""

from equinox import Module, field

import jax.numpy as jnp

from cryojax.typing import Real_


class ElectronDose(Module, strict=True):
    """Models the exposure to electrons during image formation.

    **Attributes:**

    `electrons_per_angstrom_squared`: The integrated electron flux.
    """

    electrons_per_angstrom_squared: Real_ = field(converter=jnp.asarray)
