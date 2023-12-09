"""
Scattering methods using non-uniform FFTs.
"""

from __future__ import annotations

__all__ = [
    "project_with_nufft",
    "project_atoms_with_nufft",
    "NufftScattering",
]

from typing import Any, Union

import jax.numpy as jnp
import numpy as np

from ..density import VoxelCloud, AtomCloud
from .base import ScatteringConfig
from ...core import field
from ...types import (
    ComplexImage,
    RealCloud,
    CloudCoords,
)
from ...utils import nufft


class NufftScattering(ScatteringConfig):
    """
    Scatter points to image plane using a
    non-uniform FFT.

    Attributes
    ----------
    eps : `float`
        See ``cryojax.utils.integration.nufft``
        for documentation.
    """

    eps: float = field(static=True, default=1e-6)

    def scatter(
        self, density: Union[VoxelCloud, AtomCloud], resolution: float
    ) -> ComplexImage:
        """Rasterize image with non-uniform FFTs."""
        if isinstance(density, VoxelCloud):
            return project_with_nufft(
                density.weights,
                density.coordinates,
                resolution,
                self.padded_shape,
                eps=self.eps,
            )
        elif isinstance(density, AtomCloud):
            return project_atoms_with_nufft(
                density.weights,
                density.coordinates,
                density.variances,
                density.identity,
                resolution,
                self.padded_shape,
                eps=self.eps,
            )
        else:
            raise NotImplementedError(
                "Supported density representations are VoxelCloud and AtomCloud"
            )


def project_atoms_with_nufft(
    weights,
    coordinates,
    variances,
    identity,
    resolution: float,
    shape: tuple[int, int],
    **kwargs: Any,
) -> ComplexImage:
    atom_types = jnp.unique(identity)
    img = jnp.zeros(shape, dtype=complex)
    for atom_type_i in atom_types:
        # Select the properties specific to that type of atom
        coords_i = coordinates[identity == atom_type_i]
        weights_i = weights[identity == atom_type_i]
        # kernel_i = atom_density_kernel[atom_type_i]

        # Build an
        atom_i_image = project_with_nufft(
            weights_i, coords_i, resolution, shape, **kwargs
        )

        # img += atom_i_image * kernel_i
        img += atom_i_image


def project_with_nufft(
    weights: RealCloud,
    coordinates: CloudCoords,
    resolution: float,
    shape: tuple[int, int],
    **kwargs: Any,
) -> ComplexImage:
    """
    Project and interpolate 3D volume point cloud
    onto imaging plane using a non-uniform FFT.

    See ``cryojax.utils.integration.nufft`` for more detail.

    Arguments
    ---------
    weights : shape `(N,)`
        Density point cloud.
    coordinates : shape `(N, 3)`
        Coordinate system of point cloud.
    resolution :
        The rasterization resolution.
    shape :
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    kwargs:
        Passed to ``cryojax.utils.integration.nufft``.

    Returns
    -------
    projection :
        The output image in fourier space.
    """
    weights, coordinates = jnp.asarray(weights), jnp.asarray(coordinates)
    M1, M2 = shape
    image_size = jnp.array(np.array([M1, M2]) * resolution)
    coordinates = jnp.flip(coordinates[:, :2], axis=-1)
    projection = nufft(weights, coordinates, image_size, shape, **kwargs)
    # Set zero frequency component to zero
    projection = projection.at[0, 0].set(0.0 + 0.0j)

    return projection / jnp.sqrt(M1 * M2)


"""
class IndependentAtomScatteringNufft(NufftScattering):
    '''
    Projects a pointcloud of atoms onto the imaging plane.
    In contrast to the work in project_with_nufft, here each atom is

    TODO: Typehints for atom_density_kernel
    '''

    def scatter(
        self,
        density: RealCloud,
        coordinates: CloudCoords,
        resolution: float,
        identity: IntCloud,
        atom_density_kernel,  # WHAT SHOULD THE TYPE BE HERE?
    ) -> ComplexImage:
        '''
        Projects a pointcloud of atoms onto the imaging plane.
        In contrast to the work in project_with_nufft, here each atom is

        TODO: Typehints for atom_density_kernel
        '''
        atom_types = jnp.unique(identity)

        img = jnp.zeros(self.padded_shape, dtype=jnp.complex64)
        for atom_type_i in atom_types:
            # Select the properties specific to that type of atom
            coords_i = coordinates[identity == atom_type_i]
            density_i = density[identity == atom_type_i]
            kernel_i = atom_density_kernel[atom_type_i]

            # Build an
            atom_i_image = project_with_nufft(
                density_i,
                coords_i,
                resolution,
                self.padded_shape,
                # atom_density_kernel[atom_type_i],
            )

            # img += atom_i_image * kernel_i
            img += atom_i_image
        return img
"""
