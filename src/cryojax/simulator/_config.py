"""
The image configuration and utility manager.
"""

from functools import cached_property
from typing import Any, Union, Callable, Optional, overload

from equinox import Module, field

import jax
import jax.numpy as jnp

from ..coordinates import CoordinateGrid, FrequencyGrid
from ..typing import Image, Real_, RealImage
from ..image import (
    crop_to_shape,
    pad_to_shape,
    resize_with_crop_or_pad,
    rescale_pixel_size,
    irfftn,
    rfftn,
)
from ..core import error_if_not_positive


class ImageConfig(Module, strict=True):
    """Configuration and utilities for an electron microscopy image.

    **Attributes:**

    `shape`:
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.

    `pixel_size`:
        The pixel size of the image in Angstroms.

    `padded_shape`:
        The shape of the image affter padding. This is
        set with the `pad_scale` variable during initialization.

    `pad_mode`:
        The method of image padding. By default, ``"constant"``.
        For all options, see ``jax.numpy.pad``.

    `rescale_method`:
        The interpolation method for pixel size rescaling. See
        ``jax.image.scale_and_translate`` for options.

    `wrapped_frequency_grid`:
        The fourier wavevectors in the imaging plane, wrapped in
        a `FrequencyGrid` object.

    `wrapped_padded_frequency_grid`:
        The fourier wavevectors in the imaging plane
        in the padded coordinate system, wrapped in
        a `FrequencyGrid` object.

    `wrapped_coordinate_grid`:
        The coordinates in the imaging plane, wrapped
        in a `CoordinateGrid` object.

    `wrapped_padded_coordinate_grid`:
        The coordinates in the imaging plane
        in the padded coordinate system, wrapped in a
        `CoordinateGrid` object.
    """

    shape: tuple[int, int] = field(static=True)
    pixel_size: Real_ = field(converter=error_if_not_positive)

    padded_shape: tuple[int, int] = field(static=True)
    pad_mode: Union[str, Callable] = field(static=True)
    rescale_method: str = field(static=True)

    wrapped_frequency_grid: FrequencyGrid
    wrapped_padded_frequency_grid: FrequencyGrid
    wrapped_coordinate_grid: CoordinateGrid
    wrapped_padded_coordinate_grid: CoordinateGrid

    @overload
    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: Real_,
        padded_shape: tuple[int, int],
        *,
        pad_mode: Union[str, Callable] = "constant",
        rescale_method: str = "bicubic",
    ): ...

    @overload
    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: Real_,
        *,
        pad_scale: float = 1.0,
        pad_mode: Union[str, Callable] = "constant",
        rescale_method: str = "bicubic",
    ): ...

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: Real_,
        padded_shape: Optional[tuple[int, int]] = None,
        *,
        pad_scale: float = 1.0,
        pad_mode: Union[str, Callable] = "constant",
        rescale_method: str = "bicubic",
    ):
        """**Arguments:**

        `pad_scale`: A scale factor at which to pad the image. This is
                     optionally used to set `padded_shape` and must be
                     greater than `1`.
        """
        self.shape = shape
        self.pixel_size = pixel_size
        self.pad_mode = pad_mode
        self.rescale_method = rescale_method
        # Set shape after padding
        if padded_shape is None:
            self.padded_shape = (int(pad_scale * shape[0]), int(pad_scale * shape[1]))
        else:
            self.padded_shape = padded_shape
        # Set coordinates
        self.wrapped_frequency_grid = FrequencyGrid(shape=self.shape)
        self.wrapped_padded_frequency_grid = FrequencyGrid(shape=self.padded_shape)
        self.wrapped_coordinate_grid = CoordinateGrid(shape=self.shape)
        self.wrapped_padded_coordinate_grid = CoordinateGrid(shape=self.padded_shape)

    def __check_init__(self):
        if self.padded_shape[0] < self.shape[0] or self.padded_shape[1] < self.shape[1]:
            raise AttributeError(
                f"ImageConfig.padded_shape is less than ImageConfig.shape in one or more dimensions."
            )

    @cached_property
    def wrapped_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.wrapped_coordinate_grid

    @cached_property
    def wrapped_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_frequency_grid / self.pixel_size

    @cached_property
    def wrapped_padded_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.wrapped_padded_coordinate_grid

    @cached_property
    def wrapped_padded_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_padded_frequency_grid / self.pixel_size

    def rescale_to_pixel_size(
        self, image: Image, current_pixel_size: Real_, is_real: bool = True
    ) -> RealImage:
        """Rescale the image pixel size using real-space interpolation. Only
        interpolate if the `pixel_size` is not the `current_pixel_size`."""
        if is_real:
            rescale_fn = lambda im: rescale_pixel_size(
                im, current_pixel_size, self.pixel_size, method=self.rescale_method
            )
        else:
            rescale_fn = lambda im: rfftn(
                rescale_pixel_size(
                    irfftn(im, s=self.padded_shape),
                    current_pixel_size,
                    self.pixel_size,
                    method=self.rescale_method,
                )
            )
        null_fn = lambda im: im
        return jax.lax.cond(
            jnp.isclose(current_pixel_size, self.pixel_size),
            null_fn,
            rescale_fn,
            image,
        )

    def crop_to_shape(self, image: RealImage) -> RealImage:
        """Crop an image."""
        return crop_to_shape(image, self.shape)

    def pad_to_padded_shape(self, image: RealImage, **kwargs: Any) -> RealImage:
        """Pad an image."""
        return pad_to_shape(image, self.padded_shape, mode=self.pad_mode, **kwargs)

    def crop_or_pad_to_padded_shape(self, image: RealImage, **kwargs: Any) -> RealImage:
        """Reshape an image using cropping or padding."""
        return resize_with_crop_or_pad(
            image, self.padded_shape, mode=self.pad_mode, **kwargs
        )
