"""
A type hint utility module using jaxtyping. If these public type hints
are not sufficient, add private type hints to the top of the file.
"""

__all__ = [
    "Real_",
    "Complex_",
    "Int_",
    "RealVector",
    "ComplexVector",
    "Vector",
    "RealImage",
    "ComplexImage",
    "Image",
    "RealVolume",
    "ComplexVolume",
    "Volume",
    "RealCloud",
    "ComplexCloud",
    "IntCloud",
    "Cloud",
    "CloudCoords2D",
    "CloudCoords3D",
]

from typing import Union
from jaxtyping import Array, Float, Complex, Int

# 0-d array type hints
Real_ = Float[Array, "..."]
"""Type hint for a real-valued number."""

Complex_ = Complex[Array, "..."]
"""Type hint for an integer."""

Int_ = Int[Array, "..."]
"""Type hint for an integer."""

# 1-d array type hints
RealVector = Float[Array, "... N"]
"""Type hint for a real-valued vector."""

ComplexVector = Complex[Array, "... N"]
"""Type hint for an complex-valued vector."""

Vector = Union[RealVector, ComplexVector]
"""Type hint for a vector."""

# 2-d array type hints
RealImage = Float[Array, "... N1 N2"]
"""Type hint for an real-valued image."""

ComplexImage = Complex[Array, "... N1 N2"]
"""Type hint for an complex-valued image."""

Image = Union[RealImage, ComplexImage]
"""Type hint for an image."""

ImageCoords = Float[Array, "... N1 N2 2"]
"""Type hint for a coordinate system."""

# 3-d array type hints
RealVolume = Float[Array, "... N1 N2 N3"]
"""Type hint for an real-valued volume."""

ComplexVolume = Complex[Array, "... N1 N2 N3"]
"""Type hint for an complex-valued volume."""

Volume = Union[RealVolume, ComplexVolume]
"""Type hint for an volume."""

RealCubicVolume = Float[Array, "N N N"]
"""Type hint for a real-valued cubic volume."""

ComplexCubicVolume = Complex[Array, "N N N"]
"""Type hint for a complex-valued cubic volume."""

CubicVolume = Union[RealCubicVolume, ComplexCubicVolume]
"""Type hint for a cubic volume."""

VolumeCoords = Float[Array, "... N1 N2 N3 3"]
"""Type hint for a volume coordinate system."""

VolumeSliceCoords = Float[Array, "N N//2+1 1 3"] | Float[Array, "N N 1 3"]
"""Type hint for a volume slice coordinate system."""

# Point cloud type hints (non-uniformly spaced points).
RealCloud = Float[Array, "... N"]
"""Type hint for a real-valued point cloud."""

IntCloud = Int[Array, "... N"]
"""Type hint for an integer-valued point cloud."""

ComplexCloud = Complex[Array, "... N"]
"""Type hint for a complex-valued point cloud."""

Cloud = Union[RealCloud, ComplexCloud]
"""Type hint for a point cloud."""

CloudCoords3D = Float[Array, "... N 3"]
"""Type hint for a 3D point cloud coordinate system."""

CloudCoords2D = Float[Array, "... N 2"]
"""Type hint for a 2D point cloud coordinate system."""
