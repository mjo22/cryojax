"""
A type hint utility module using jaxtyping. If these public type hints
are not sufficient, use jaxtyping directly.
"""

import jaxtyping as jt

# 0-d array type hints
Real_ = jt.Float[jt.Array, ""]
"""Type hint for a real-valued number."""

Complex_ = jt.Complex[jt.Array, ""]
"""Type hint for an integer."""

Int_ = jt.Int[jt.Array, ""]
"""Type hint for an integer."""

# 1-d array type hints
RealVector = jt.Float[jt.Array, "N"]
"""Type hint for a real-valued vector."""

ComplexVector = jt.Complex[jt.Array, "N"]
"""Type hint for an complex-valued vector."""

Vector = jt.Inexact[jt.Array, "N"]
"""Type hint for a vector."""

# 2-d array type hints
RealImage = jt.Float[jt.Array, "N1 N2"]
"""Type hint for an real-valued image."""

ComplexImage = jt.Complex[jt.Array, "N1 N2"]
"""Type hint for an complex-valued image."""

Image = jt.Inexact[jt.Array, "N1 N2"]
"""Type hint for an image."""

ImageCoords = jt.Float[jt.Array, "N1 N2 2"]
"""Type hint for a coordinate system."""

# 3-d array type hints
RealVolume = jt.Float[jt.Array, "N1 N2 N3"]
"""Type hint for an real-valued volume."""

ComplexVolume = jt.Complex[jt.Array, "N1 N2 N3"]
"""Type hint for an complex-valued volume."""

Volume = jt.Inexact[jt.Array, "N1 N2 N3"]
"""Type hint for an volume."""

RealCubicVolume = jt.Float[jt.Array, "N N N"]
"""Type hint for a real-valued cubic volume."""

ComplexCubicVolume = jt.Complex[jt.Array, "N N N"]
"""Type hint for a complex-valued cubic volume."""

CubicVolume = jt.Inexact[jt.Array, "N N N"]
"""Type hint for a cubic volume."""

VolumeCoords = jt.Float[jt.Array, "N1 N2 N3 3"]
"""Type hint for a volume coordinate system."""

VolumeSliceCoords = jt.Float[jt.Array, "N1 N2 1 3"]
"""Type hint for a volume slice coordinate system."""

# Point cloud type hints (non-uniformly spaced points).
RealCloud = jt.Float[jt.Array, "N"]
"""Type hint for a real-valued point cloud."""

IntCloud = jt.Int[jt.Array, "N"]
"""Type hint for an integer-valued point cloud."""

CloudCoords3D = jt.Float[jt.Array, "N 3"]
"""Type hint for a 3D point cloud coordinate system."""

CloudCoords2D = jt.Float[jt.Array, "N 2"]
"""Type hint for a 2D point cloud coordinate system."""

del jt
