import abc
import pathlib
from typing import Generic, Literal, Optional, TypeVar

import numpy as np
from jaxtyping import Float, Int, PyTree

from ...internal import NDArrayLike
from .._dataset import AbstractDataset


T = TypeVar("T")


class AbstractParticleParameterFile(AbstractDataset[T], Generic[T]):
    @abc.abstractmethod
    def __setitem__(self, index, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path_to_output(self) -> pathlib.Path:
        raise NotImplementedError

    @path_to_output.setter
    @abc.abstractmethod
    def path_to_output(self, value: str | pathlib.Path):
        raise NotImplementedError

    @property
    def mode(self) -> Literal["r", "w"]:
        raise NotImplementedError


class AbstractParticleStackDataset(AbstractDataset[T], Generic[T]):
    @property
    @abc.abstractmethod
    def parameter_file(self) -> AbstractParticleParameterFile:
        raise NotImplementedError

    @abc.abstractmethod
    def __setitem__(self, index, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, value: T):
        raise NotImplementedError

    @abc.abstractmethod
    def write_images(
        self,
        index_array: Int[np.ndarray, " _"],
        images: Float[NDArrayLike, "... _ _"],
        parameters: Optional[PyTree] = None,
    ):
        raise NotImplementedError

    @property
    def mode(self) -> Literal["r", "w"]:
        raise NotImplementedError
