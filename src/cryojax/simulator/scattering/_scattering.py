"""
Routines to model image formation from 3D electron density
fields.
"""

from __future__ import annotations

__all__ = ["ScatteringConfig"]

from abc import abstractmethod
from typing import Any

from ..density import ElectronDensity
from ..pose import Pose

from ...core import field, Module
from ...typing import (
    Real_,
    RealImage,
    ComplexImage,
    ImageCoords,
)
from ...utils import (
    make_frequencies,
    make_coordinates,
    crop,
    pad,
    resize,
)


class ScatteringConfig(Module):
    """
    Configuration for an electron microscopy image with a
    particular scattering method.

    In subclasses, overwrite the ``ScatteringConfig.scatter``
    routine.

    Attributes
    ----------
    shape :
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pad_scale :
        The scale at which to pad (or upsample) the image
        when computing it in the object plane. This
        should be a floating point number greater than
        or equal to 1. By default, it is 1 (no padding).
    freqs :
        The fourier wavevectors in the imaging plane.
    padded_freqs :
        The fourier wavevectors in the imaging plane
        in the padded coordinate system.
    coords :
        The coordinates in the imaging plane.
    padded_coords :
        The coordinates in the imaging plane
        in the padded coordinate system.
    """

    shape: tuple[int, int] = field(static=True)
    pad_scale: float = field(static=True, default=1.0)

    padded_shape: tuple[int, int] = field(static=True, init=False)

    freqs: ImageCoords = field(static=True, init=False)
    padded_freqs: ImageCoords = field(static=True, init=False)
    coords: ImageCoords = field(static=True, init=False)
    padded_coords: ImageCoords = field(static=True, init=False)

    def __post_init__(self):
        # Set shape after padding
        padded_shape = tuple([int(s * self.pad_scale) for s in self.shape])
        self.padded_shape = padded_shape
        # Set coordinates
        self.freqs = make_frequencies(self.shape)
        self.padded_freqs = make_frequencies(self.padded_shape)
        self.coords = make_coordinates(self.shape)
        self.padded_coords = make_coordinates(self.padded_shape)

    @abstractmethod
    def scatter(
        self, density: ElectronDensity, pose: Pose, resolution: Real_, 
    ) -> ComplexImage:
        """
        Compute the scattered wave of the electron
        density in the exit plane.

        Arguments
        ---------
        density :
            The electron density representation.
        pose :
            The imaging pose.
        resolution :
            The rasterization resolution.
        """
        raise NotImplementedError

    def crop(self, image: RealImage) -> RealImage:
        """Crop an image."""
        return crop(image, self.shape)

    def pad(self, image: RealImage, **kwargs: Any) -> RealImage:
        """Pad an image."""
        return pad(image, self.padded_shape, **kwargs)

    def downsample(
        self, image: ComplexImage, method="lanczos5", **kwargs: Any
    ) -> ComplexImage:
        """Downsample an image."""
        return resize(
            image, self.shape, antialias=False, method=method, **kwargs
        )

    def upsample(
        self, image: ComplexImage, method="bicubic", **kwargs: Any
    ) -> ComplexImage:
        """Upsample an image."""
        return resize(image, self.padded_shape, method=method, **kwargs)
