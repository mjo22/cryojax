"""
Type hints.
"""

__all__ = [
    "Real_",
    "Complex_",
    "Integer_",
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
    "Cloud",
    "CloudCoords",
]

from typing import Union
from jaxtyping import Array, Float, Complex, Int

# 0-d array type hints
Real_ = Float[Array, ""]
"""Type hint for a real-valued number."""

Complex_ = Complex[Array, ""]
"""Type hint for an integer."""

Integer_ = Int[Array, ""]
"""Type hint for an integer."""

# 1-d array type hints
RealVector = Float[Array, "N"]
"""Type hint for a real-valued vector."""

ComplexVector = Complex[Array, "N"]
"""Type hint for an complex-valued vector."""

Vector = Union[RealVector, ComplexVector]
"""Type hint for a vector."""

# 2-d array type hints
RealImage = Float[Array, "N1 N2"]
"""Type hint for an real-valued image."""

ComplexImage = Complex[Array, "N1 N2"]
"""Type hint for an complex-valued image."""

Image = Union[RealImage, ComplexImage]
"""Type hint for an image."""

ImageCoords = Float[Array, "N1 N2 2"]
"""Type hint for a coordinate system."""

# 3-d array type hints
RealVolume = Float[Array, "N1 N2 N3"]
"""Type hint for an real-valued volume."""

ComplexVolume = Complex[Array, "N1 N2 N3"]
"""Type hint for an complex-valued volume."""

Volume = Union[RealVolume, ComplexVolume]
"""Type hint for an volume."""

VolumeCoords = Float[Array, "N1 N2 N3 3"]
"""Type hint for a volume coordinate system."""

# 3D Point cloud type hints (non-uniformly spaced points).
RealCloud = Float[Array, "N"]
"""Type hint for a real-valued point cloud."""

IntCloud = Int[Array, "N"]
"""Type hint for an integer-valued point cloud."""

ComplexCloud = Complex[Array, "N"]
"""Type hint for a complex-valued point cloud."""

Cloud = Union[RealCloud, ComplexCloud]
"""Type hint for a point cloud."""

CloudCoords = Float[Array, "N 3"]
"""Type hint for a point cloud coordinate system."""
