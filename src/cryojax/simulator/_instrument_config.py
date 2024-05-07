"""
The image configuration and utility manager.
"""

import math
from functools import cached_property
from typing import Any, Callable, Optional, Union

import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Float

from .._errors import error_if_not_positive
from ..constants import convert_keV_to_angstroms
from ..coordinates import CoordinateGrid, FrequencyGrid
from ..image import (
    crop_to_shape,
    pad_to_shape,
    resize_with_crop_or_pad,
)


class InstrumentConfig(Module, strict=True):
    """Configuration and utilities for an electron microscopy image."""

    shape: tuple[int, int]
    pixel_size: Float[Array, ""]
    voltage_in_kilovolts: Float[Array, ""]
    electrons_per_angstrom_squared: Float[Array, ""]

    padded_shape: tuple[int, int]
    pad_mode: Union[str, Callable]

    def __init__(
        self,
        shape: tuple[int, int],
        pixel_size: float | Float[Array, ""],
        voltage_in_kilovolts: float | Float[Array, ""],
        electrons_per_angstrom_squared: float | Float[Array, ""] = 100.0,
        padded_shape: Optional[tuple[int, int]] = None,
        *,
        pad_scale: float = 1.0,
        pad_mode: Union[str, Callable] = "constant",
    ):
        """**Arguments:**

        - `shape`:
            Shape of the imaging plane in pixels.
            ``width, height = shape[0], shape[1]``
            is the size of the desired imaging plane.
        - `pixel_size`:
            The pixel size of the image in Angstroms.
        - `padded_shape`:
            The shape of the image affter padding. This is
            set with the `pad_scale` variable during initialization.
        - `pad_scale`: A scale factor at which to pad the image. This is
                       optionally used to set `padded_shape` and must be
                       greater than `1`. If `padded_shape` is set, this
                       argument is ignored.
        - `pad_mode`:
            The method of image padding. By default, ``"constant"``.
            For all options, see ``jax.numpy.pad``.
        """
        self.shape = shape
        self.pixel_size = error_if_not_positive(jnp.asarray(pixel_size))
        self.voltage_in_kilovolts = error_if_not_positive(
            jnp.asarray(voltage_in_kilovolts)
        )
        self.electrons_per_angstrom_squared = error_if_not_positive(
            jnp.asarray(electrons_per_angstrom_squared)
        )
        self.pad_mode = pad_mode
        # Set shape after padding
        if padded_shape is None:
            self.padded_shape = (int(pad_scale * shape[0]), int(pad_scale * shape[1]))
        else:
            self.padded_shape = padded_shape

    def __check_init__(self):
        if self.padded_shape[0] < self.shape[0] or self.padded_shape[1] < self.shape[1]:
            raise AttributeError(
                "ImageConfig.padded_shape is less than ImageConfig.shape in one or "
                "more dimensions."
            )

    @property
    def wavelength_in_angstroms(self) -> Float[Array, ""]:
        return convert_keV_to_angstroms(self.voltage_in_kilovolts)

    @cached_property
    def wrapped_coordinate_grid_in_pixels(self) -> CoordinateGrid:
        return CoordinateGrid(shape=self.shape)

    @cached_property
    def wrapped_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.wrapped_coordinate_grid_in_pixels  # type: ignore

    @cached_property
    def wrapped_frequency_grid_in_pixels(self) -> FrequencyGrid:
        return FrequencyGrid(shape=self.shape)

    @cached_property
    def wrapped_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_frequency_grid_in_pixels / self.pixel_size

    @cached_property
    def wrapped_full_frequency_grid_in_pixels(self) -> FrequencyGrid:
        return FrequencyGrid(shape=self.shape, half_space=False)

    @cached_property
    def wrapped_full_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_full_frequency_grid_in_pixels / self.pixel_size

    @cached_property
    def wrapped_padded_coordinate_grid_in_pixels(self) -> CoordinateGrid:
        return CoordinateGrid(shape=self.padded_shape)

    @cached_property
    def wrapped_padded_coordinate_grid_in_angstroms(self) -> CoordinateGrid:
        return self.pixel_size * self.wrapped_padded_coordinate_grid_in_pixels  # type: ignore

    @cached_property
    def wrapped_padded_frequency_grid_in_pixels(self) -> FrequencyGrid:
        return FrequencyGrid(shape=self.padded_shape)

    @cached_property
    def wrapped_padded_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_padded_frequency_grid_in_pixels / self.pixel_size

    @cached_property
    def wrapped_padded_full_frequency_grid_in_pixels(self) -> FrequencyGrid:
        return FrequencyGrid(shape=self.padded_shape, half_space=False)

    @cached_property
    def wrapped_padded_full_frequency_grid_in_angstroms(self) -> FrequencyGrid:
        return self.wrapped_padded_full_frequency_grid_in_pixels / self.pixel_size

    def crop_to_shape(
        self, image: Float[Array, "y_dim x_dim"]
    ) -> Float[Array, "{self.y_dim} {self.x_dim}"]:
        """Crop an image."""
        return crop_to_shape(image, self.shape)

    def pad_to_padded_shape(
        self, image: Float[Array, "y_dim x_dim"], **kwargs: Any
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim}"]:
        """Pad an image."""
        return pad_to_shape(image, self.padded_shape, mode=self.pad_mode, **kwargs)

    def crop_or_pad_to_padded_shape(
        self, image: Float[Array, "y_dim x_dim"], **kwargs: Any
    ) -> Float[Array, "{self.padded_y_dim} {self.padded_x_dim}"]:
        """Reshape an image using cropping or padding."""
        return resize_with_crop_or_pad(
            image, self.padded_shape, mode=self.pad_mode, **kwargs
        )

    @property
    def n_pixels(self) -> int:
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
    def padded_n_pixels(self) -> int:
        return math.prod(self.padded_shape)
