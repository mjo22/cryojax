"""
Models the electron dose.
"""

from equinox import field, Module
from jaxtyping import Shaped

from ..core import error_if_not_positive
from ..typing import RealNumber


class ElectronDose(Module, strict=True):
    """Models the exposure to electrons during image formation.

    **Attributes:**

    `electrons_per_angstrom_squared`: The integrated electron flux.
    """

    electrons_per_angstrom_squared: Shaped[RealNumber, "..."] = field(
        converter=error_if_not_positive
    )
