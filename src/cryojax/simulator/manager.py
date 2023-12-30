"""
The image configuration and utility manager.
"""

from __future__ import annotations

__all__ = ["ImageManager"]

from typing import Any, Union, Callable

import jax.numpy as jnp

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
    crop_or_pad,
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
    pad_mode :
        The method of image padding. By default, ``"edge"``.
        For all options, see ``jax.numpy.pad``.
    frequency_grid :
        The fourier wavevectors in the imaging plane.
    padded_frequency_grid :
        The fourier wavevectors in the imaging plane
        in the padded coordinate system.
    coordinate_grid :
        The coordinates in the imaging plane.
    padded_coordinate_grid :
        The coordinates in the imaging plane
        in the padded coordinate system.
    """

    shape: tuple[int, int] = field(static=True)
    pad_scale: float = field(static=True, default=1.0)
    pad_mode: Union[str, Callable] = field(static=True, default="edge")

    padded_shape: tuple[int, int] = field(static=True, init=False)

    frequency_grid: ImageCoords = field(init=False)
    padded_frequency_grid: ImageCoords = field(init=False)
    coordinate_grid: ImageCoords = field(init=False)
    padded_coordinate_grid: ImageCoords = field(init=False)

    def __post_init__(self):
        # Set shape after padding
        padded_shape = tuple([int(s * self.pad_scale) for s in self.shape])
        self.padded_shape = padded_shape
        # Set coordinates
        self.frequency_grid = make_frequencies(self.shape)
        self.padded_frequency_grid = make_frequencies(self.padded_shape)
        self.coordinate_grid = make_coordinates(self.shape)
        self.padded_coordinate_grid = make_coordinates(self.padded_shape)

    def crop_to_shape(self, image: Image) -> Image:
        """Crop an image."""
        return crop(image, self.shape)

    def pad_to_padded_shape(self, image: Image, **kwargs: Any) -> Image:
        """Pad an image."""
        return pad(image, self.padded_shape, mode=self.pad_mode, **kwargs)

    def crop_or_pad_to_padded_shape(
        self, image: Image, **kwargs: Any
    ) -> Image:
        """Reshape an image using cropping or padding."""
        return crop_or_pad(
            image, self.padded_shape, mode=self.pad_mode, **kwargs
        )

    def normalize_to_cistem(
        self, image: Image, is_real: bool = False
    ) -> Image:
        """Normalize images on the exit plane according to cisTEM conventions."""
        M1, M2 = image.shape
        if is_real:
            raise NotImplementedError(
                "Normalization to cisTEM conventions not supported for real input."
            )
        else:
            # Set zero frequency component to zero
            image = image.at[0, 0].set(0.0 + 0.0j)
            # cisTEM normalization convention for projections
            return image / jnp.sqrt(M1 * M2)
