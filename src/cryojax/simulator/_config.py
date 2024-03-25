"""
The image configuration and utility manager.
"""

import math
from functools import cached_property
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from equinox import field, Module
from jaxtyping import Array, Complex, Float, Shaped

from ..coordinates import CoordinateGrid, FrequencyGrid
from ..core import error_if_not_positive
from ..image import (
    crop_to_shape,
    irfftn,
    pad_to_shape,
    rescale_pixel_size,
    resize_with_crop_or_pad,
    rfftn,
)
from ..typing import RealImage, RealNumber


class ImageConfig(Module, strict=True):
    """Configuration and utilities for an electron microscopy image.

    **Attributes:**

    - `shape`:
        Shape of the imaging plane in pixels.
        ``width, height = shape[0], shape[1]``
        is the size of the desired imaging plane.
    - `pixel_size`:
        The pixel size of the image in Angstroms.
    - `padded_shape`:
        The shape of the image affter padding. This is
        set with the `pad_scale` variable during initialization.
    - `pad_mode`:
        The method of image padding. By default, ``"constant"``.
        For all options, see ``jax.numpy.pad``.
    - `rescale_method`:
        The interpolation method for pixel size rescaling. See
        ``jax.image.scale_and_translate`` for options.
    - `wrapped_frequency_grid_in_pixels`:
        The fourier wavevectors in the imaging plane, wrapped in
        a `FrequencyGrid` object.
    - `wrapped_padded_frequency_grid_in_pixels`:
        The fourier wavevectors in the imaging plane
        in the padded coordinate system, wrapped in
        a `FrequencyGrid` object.
    - `wrapped_coordinate_grid_in_pixels`:
        The coordinates in the imaging plane, wrapped
        in a `CoordinateGrid` object.
    - `wrapped_padded_coordinate_grid_in_pixels`:
        The coordinates in the imaging plane
        in the padded coordinate system, wrapped in a
        `CoordinateGrid` object.
    """

    shape: tuple[int, int] = field(static=True)
    pixel_size: Shaped[RealNumber, "..."] = field(converter=error_if_not_positive)

    padded_shape: tuple[int, int] = field(static=True)
    pad_mode: Union[str, Callable] = field(static=True)
    rescale_method: str = field(static=True)

    wrapped_frequency_grid_in_pixels: FrequencyGrid
    wrapped_padded_frequency_grid_in_pixels: FrequencyGrid
    wrapped_coordinate_grid_in_pixels: CoordinateGrid
    wrapped_padded_coordinate_grid_in_pixels: CoordinateGrid

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: float | Shaped[RealNumber, "..."],
        padded_shape: Optional[tuple[int, int]] = None,
        *,
        pad_scale: float = 1.0,
        pad_mode: Union[str, Callable] = "constant",
        rescale_method: str = "bicubic",
    ):
        """**Arguments:**

        - `pad_scale`: A scale factor at which to pad the image. This is
                       optionally used to set `padded_shape` and must be
                       greater than `1`. If `padded_shape` is set, this
                       argument is ignored.
        """
        self.shape = shape
        self.pixel_size = jnp.asarray(pixel_size)
        self.pad_mode = pad_mode
        self.rescale_method = rescale_method
        # Set shape after padding
        if padded_shape is None:
            self.padded_shape = (int(pad_scale * shape[0]), int(pad_scale * shape[1]))
        else:
            self.padded_shape = padded_shape
        # Set coordinates
        self.wrapped_frequency_grid_in_pixels = FrequencyGrid(shape=self.shape)
        self.wrapped_padded_frequency_grid_in_pixels = FrequencyGrid(
            shape=self.padded_shape
        )
        self.wrapped_coordinate_grid_in_pixels = CoordinateGrid(shape=self.shape)
        self.wrapped_padded_coordinate_grid_in_pixels = CoordinateGrid(
            shape=self.padded_shape
        )

    def __check_init__(self):
        if self.padded_shape[0] < self.shape[0] or self.padded_shape[1] < self.shape[1]:
            raise AttributeError(
                "ImageConfig.padded_shape is less than ImageConfig.shape in one or "
                "more dimensions."
            )

    @cached_property
    def wrapped_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.wrapped_coordinate_grid_in_pixels  # type: ignore

    @cached_property
    def wrapped_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_frequency_grid_in_pixels / self.pixel_size

    @cached_property
    def wrapped_padded_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.wrapped_padded_coordinate_grid_in_pixels  # type: ignore

    @cached_property
    def wrapped_padded_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_padded_frequency_grid_in_pixels / self.pixel_size

    def rescale_to_pixel_size(
        self,
        real_or_fourier_image: (
            Float[Array, "{self.padded_y_dim} {self.padded_x_dim}"]
            | Complex[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1}"]
        ),
        current_pixel_size: RealNumber,
        is_real: bool = True,
    ) -> Complex[Array, "{self.padded_y_dim} {self.padded_x_dim//2+1}"]:
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
            real_or_fourier_image,
        )

    def crop_to_shape(
        self, image: RealImage
    ) -> Float[Array, "{self.y_dim} {self.x_dim}"]:
        """Crop an image."""
        return crop_to_shape(image, self.shape)

    def pad_to_padded_shape(
        self, image: RealImage, **kwargs: Any
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim}"]:
        """Pad an image."""
        return pad_to_shape(image, self.padded_shape, mode=self.pad_mode, **kwargs)

    def crop_or_pad_to_padded_shape(
        self, image: RealImage, **kwargs: Any
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim}"]:
        """Reshape an image using cropping or padding."""
        return resize_with_crop_or_pad(
            image, self.padded_shape, mode=self.pad_mode, **kwargs
        )

    @property
    def n_pix(self) -> int:
        return math.prod(self.shape)

    @property
    def y_dim(self) -> int:
        return self.shape[0]

    @property
    def x_dim(self) -> int:
        return self.shape[1]

    @property
    def padded_y_dim(self) -> int:
        return self.padded_shape[0]

    @property
    def padded_x_dim(self) -> int:
        return self.padded_shape[1]

    @property
    def padded_n_pix(self) -> int:
        return math.prod(self.padded_shape)
