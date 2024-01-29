"""
The image configuration and utility manager.
"""

from functools import cached_property
from typing import Any, Union, Callable, Optional

from equinox import Module, field

import jax.numpy as jnp

from ..coordinates import CoordinateGrid, FrequencyGrid
from ..typing import Image, Real_, RealImage
from ..image import (
    crop_to_shape,
    pad_to_shape,
    resize_with_crop_or_pad,
    normalize_image,
    rescale_pixel_size,
)


class ImageManager(Module, strict=True):
    """
    Configuration and utilities for an electron microscopy image.

    Attributes
    ----------
    shape :
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    pixel_size :
        The pixel size of the image in Angstroms.
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
    pixel_size: Real_ = field(converter=jnp.asarray)

    pad_scale: float = field(static=True)
    pad_mode: Union[str, Callable] = field(static=True)
    rescale_method: str = field(static=True)

    padded_shape: tuple[int, int] = field(static=True)

    frequency_grid: FrequencyGrid
    padded_frequency_grid: FrequencyGrid
    coordinate_grid: CoordinateGrid
    padded_coordinate_grid: CoordinateGrid

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: Real_,
        *,
        pad_scale: float = 1.0,
        pad_mode: Union[str, Callable] = "constant",
        rescale_method: str = "bicubic"
    ):
        self.shape = shape
        self.pixel_size = pixel_size
        self.pad_scale = pad_scale
        self.pad_mode = pad_mode
        self.rescale_method = rescale_method
        # Set shape after padding
        padded_shape = tuple([int(s * self.pad_scale) for s in self.shape])
        self.padded_shape = padded_shape
        # Set coordinates
        self.frequency_grid = FrequencyGrid(shape=self.shape)
        self.padded_frequency_grid = FrequencyGrid(shape=self.padded_shape)
        self.coordinate_grid = CoordinateGrid(shape=self.shape)
        self.padded_coordinate_grid = CoordinateGrid(shape=self.padded_shape)

    @cached_property
    def coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.coordinate_grid

    @cached_property
    def frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.frequency_grid / self.pixel_size

    @cached_property
    def padded_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.padded_coordinate_grid

    @cached_property
    def padded_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.padded_frequency_grid / self.pixel_size

    def rescale_to_pixel_size(
        self, image: RealImage, current_pixel_size: Real_
    ) -> RealImage:
        """Rescale the image pixel size."""
        return rescale_pixel_size(
            image,
            current_pixel_size,
            self.pixel_size,
            method=self.rescale_method,
        )

    def crop_to_shape(self, image: Image) -> Image:
        """Crop an image."""
        return crop_to_shape(image, self.shape)

    def pad_to_padded_shape(self, image: Image, **kwargs: Any) -> Image:
        """Pad an image."""
        return pad_to_shape(
            image, self.padded_shape, mode=self.pad_mode, **kwargs
        )

    def crop_or_pad_to_padded_shape(
        self, image: Image, **kwargs: Any
    ) -> Image:
        """Reshape an image using cropping or padding."""
        return resize_with_crop_or_pad(
            image, self.padded_shape, mode=self.pad_mode, **kwargs
        )

    def normalize_image(
        self,
        image: Image,
        is_real: bool = True,
        half_space: bool = True,
        shape_in_real_space: Optional[tuple[int, int]] = None,
    ) -> Image:
        """Normalize an image so that it is mean 0 and standard deviation 1 in real space."""
        return normalize_image(
            image,
            is_real=is_real,
            half_space=half_space,
            shape_in_real_space=shape_in_real_space,
        )
