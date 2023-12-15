__all__ = ["Ellipsoid"]


from ._density import ElectronDensity
from ..pose import Pose
from ...typing import Real_
from ...core import field

class Ellipsoid(ElectronDensity):
    axis_a: Real_ = field() # revisit when break symmetry (large/med/small, ratio, etc.)
    axis_b: Real_ = field()
    axis_c: Real_ = field()

    def view(self, pose: Pose) -> "Ellipsoid":
        return self # TODO: revisit this choice... impicit equation for platonic shapes?

