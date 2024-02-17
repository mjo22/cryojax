"""
Models the electron dose.
"""

from equinox import Module

from cryojax.image.operators import RealOperatorLike


class ElectronDose(Module, strict=True):
    """Models the exposure to electrons during image formation.

    **Attributes:**

    `electrons_per_angstrom_squared`: The integrated electron flux a function of
                                      spatial coordinate. Typically, this
                                      is just set to be `Constant(...)`.
    """

    electrons_per_angstrom_squared: RealOperatorLike

    def __init__(self, electrons_per_angstrom_squared: RealOperatorLike):
        self.electrons_per_angstrom_squared = electrons_per_angstrom_squared
