"""
The image configuration and utility manager.
"""

from __future__ import annotations

__all__ = ["ImageManager"]

from typing import Any

from ..core import field, Buffer
from ..typing import (
    Image,
    ImageCoords,
)
from ..utils import (
    make_frequencies,
    make_coordinates,
    crop,
    pad,
    resize,
)


class ImageManager(Buffer):
    """
    Configuration and utilities for an electron microscopy image.

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

    freqs: ImageCoords = field(init=False)
    padded_freqs: ImageCoords = field(init=False)
    coords: ImageCoords = field(init=False)
    padded_coords: ImageCoords = field(init=False)

    def __post_init__(self):
        # Set shape after padding
        padded_shape = tuple([int(s * self.pad_scale) for s in self.shape])
        self.padded_shape = padded_shape
        # Set coordinates
        self.freqs = make_frequencies(self.shape)
        self.padded_freqs = make_frequencies(self.padded_shape)
        self.coords = make_coordinates(self.shape)
        self.padded_coords = make_coordinates(self.padded_shape)

    def crop(self, image: Image) -> Image:
        """Crop an image."""
        return crop(image, self.shape)

    def pad(self, image: Image, **kwargs: Any) -> Image:
        """Pad an image."""
        return pad(image, self.padded_shape, **kwargs)

    def downsample(
        self, image: Image, method="lanczos5", **kwargs: Any
    ) -> Image:
        """Downsample an image."""
        return resize(
            image, self.shape, antialias=False, method=method, **kwargs
        )
