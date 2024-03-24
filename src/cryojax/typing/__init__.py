"""
A type hint utility module using jaxtyping. This module is primarily used to
enforce / communicate conventions across `cryojax`.
"""

import jaxtyping as jt


# 0-d array type hints
RealNumber = jt.Float[jt.Array, ""]
"""Type alias for `jaxtyping.Float[jax.Array, ""]`."""

ComplexNumber = jt.Complex[jt.Array, ""]
"""Type alias for `jaxtyping.Complex[jax.Array, ""]`."""

Integer = jt.Int[jt.Array, ""]
"""Type alias for `jaxtyping.Int[jax.Array, ""]`."""

# 2-d array type hints
RealImage = jt.Float[jt.Array, "Ny Nx"]
"""Type alias for `jaxtyping.Float[jax.Array, "Ny Nx"]`."""

ComplexImage = jt.Complex[jt.Array, "Ny Nx"]
"""Type alias for `jaxtyping.Complex[jax.Array, "Ny Nx"]`."""

Image = jt.Inexact[jt.Array, "Ny Nx"]
"""Type alias for `jaxtyping.Complex[jax.Array, "Ny Nx"]`."""

ImageCoords = jt.Float[jt.Array, "Ny Nx 2"]
"""Type alias for `jaxtyping.Float[jax.Array, "Ny Nx 2"]`."""

# 3-d array type hints
RealVolume = jt.Float[jt.Array, "Nz Ny Nx"]
"""Type alias for `jaxtyping.Float[jax.Array, "Nz Ny Nx"]`."""

ComplexVolume = jt.Complex[jt.Array, "Nz Ny Nx"]
"""Type alias for `jaxtyping.Complex[jax.Array, "Nz Ny Nx"]`."""

Volume = jt.Inexact[jt.Array, "Nz Ny Nx"]
"""Type alias for `jaxtyping.Inexact[jax.Array, "Nz Ny Nx"]`."""

RealCubicVolume = jt.Float[jt.Array, "N N N"]
"""Type alias for `jaxtyping.Float[jax.Array, "N N N"]`."""

ComplexCubicVolume = jt.Complex[jt.Array, "N N N"]
"""Type alias for `jaxtyping.Complex[jax.Array, "N N N"]`."""

CubicVolume = jt.Inexact[jt.Array, "N N N"]
"""Type alias for `jaxtyping.Inexact[jax.Array, "N N N"]`."""

VolumeCoords = jt.Float[jt.Array, "Nz Ny Nx 3"]
"""Type alias for `jaxtyping.Float[jax.Array, "Nz Ny Nx 3"]`."""

VolumeSliceCoords = jt.Float[jt.Array, "1 N N 3"]
"""Type alias for `jaxtyping.Float[jax.Array, "1 Ny Nx 3"]`."""

# Point cloud type hints (non-uniformly spaced points).
RealPointCloud = jt.Float[jt.Array, " N"]
"""Type alias for `jaxtyping.Float[jax.Array, " N"]`."""

IntegerPointCloud = jt.Int[jt.Array, " N"]
"""Type alias for `jaxtyping.Int[jax.Array, " N"]`."""

PointCloudCoords3D = jt.Float[jt.Array, "N 3"]
"""Type alias for `jaxtyping.Float[jax.Array, "N 3"]`."""

PointCloudCoords2D = jt.Float[jt.Array, "N 2"]
"""Type alias for `jaxtyping.Float[jax.Array, "N 2"]`."""

del jt
