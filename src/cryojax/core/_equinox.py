"""
Add cryojax utilities onto standard equinox functionality
"""

__all__ = ["is_not_coordinate_array", "is_coordinates"]

from typing import Any

import equinox as eqx

from ._coordinates import Coordinates

#
# Filter functions
#


def is_not_coordinate_array(element: Any) -> bool:
    """Returns `True`"""
    if isinstance(element, Coordinates):
        return False
    else:
        return eqx.is_array(element)


def is_coordinates(element: Any) -> bool:
    """Returns ``True`` if ``element`` is ``Coordinates``"""
    return isinstance(element, Coordinates)
