__all__ = ["Ellipsoid"]


from ._density import ElectronDensity

class Ellipsoid(ElectronDensity):
    axis_a: float
    axis_b: float
    axis_c: float

