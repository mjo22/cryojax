"""
Models the electron dose.
"""

from equinox import Module, field

from ..typing import Real_
from ..core import error_if_not_positive


class ElectronDose(Module, strict=True):
    """Models the exposure to electrons during image formation.

    **Attributes:**

    `electrons_per_angstrom_squared`: The integrated electron flux.
    """

    electrons_per_angstrom_squared: Real_ = field(converter=error_if_not_positive)
