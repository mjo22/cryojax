""" """

from abc import abstractmethod
from typing import Any, Callable, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Shaped

from .custom_types import Out, SearchSolution, SearchState, Y as Y


class AbstractGridSearchMethod(
    eqx.Module, Generic[Out, SearchState, SearchSolution], strict=True
):
    @abstractmethod
    def init(
        self,
        fn: Callable[[PyTree[Shaped[Array, "..."] | None, "Y"], Any], Out],
        grid: PyTree[Shaped[Array, "dim ..."] | None, "Y"],
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> SearchState:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        fn: Callable[[PyTree[Shaped[Array, "..."] | None, "Y"], Any], Out],
        grid_point: PyTree[Shaped[Array, "..."] | None, "Y"],
        args: Any,
        state: SearchState,
    ) -> SearchState:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        fn: Callable[[PyTree[Shaped[Array, "..."] | None, "Y"], Any], Out],
        grid: PyTree[Shaped[Array, "dim ..."] | None, "Y"],
        args: Any,
        state: SearchState,
    ) -> SearchSolution:
        raise NotImplementedError


class MinimumState(eqx.Module, strict=True):
    current_minimum_value: Array


class MinimumSolution(eqx.Module, strict=True):
    final_state: MinimumState


class SearchForMinimum(
    AbstractGridSearchMethod[Out, MinimumState, MinimumSolution], strict=True
):
    def init(
        self,
        fn: Callable[[PyTree[Shaped[Array, "..."] | None, "Y"], Any], Out],
        grid: PyTree[Shaped[Array, "dim ..."] | None, "Y"],
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> MinimumState:
        state = MinimumState(current_minimum_value=jnp.full(f_struct.shape, -jnp.inf))
        return state

    def update(
        self,
        fn: Callable[[PyTree[Shaped[Array, "..."] | None, "Y"], Any], Out],
        grid_point: PyTree[Shaped[Array, "..."] | None, "Y"],
        args: Any,
        state: MinimumState,
    ) -> MinimumState:
        raise NotImplementedError

    def postprocess(
        self,
        fn: Callable[[PyTree[Shaped[Array, "..."] | None, "Y"], Any], Out],
        grid: PyTree[Shaped[Array, "dim ..."] | None, "Y"],
        args: Any,
        state: MinimumState,
    ) -> MinimumSolution:
        return MinimumSolution(final_state=state)
