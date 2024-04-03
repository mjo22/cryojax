"""An interface for a grid search method."""

from abc import abstractmethod
from typing import Any, Callable, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .custom_types import Grid, GridPoint, SearchSolution, SearchState


class AbstractGridSearchMethod(
    eqx.Module, Generic[SearchState, SearchSolution], strict=True
):
    @abstractmethod
    def init(
        self,
        fn: Callable[[GridPoint, Any], Array],
        tree_grid: Grid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> SearchState:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        fn: Callable[[GridPoint, Any], Array],
        tree_grid_point: GridPoint,
        args: Any,
        state: SearchState,
    ) -> SearchState:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        fn: Callable[[GridPoint, Any], Array],
        tree_grid: Grid,
        args: Any,
        state: SearchState,
    ) -> SearchSolution:
        raise NotImplementedError


class MinimumState(eqx.Module, strict=True):
    current_minimum_value: Array


class MinimumSolution(eqx.Module, strict=True):
    final_state: MinimumState


class SearchForMinimum(
    AbstractGridSearchMethod[MinimumState, MinimumSolution], strict=True
):
    def init(
        self,
        fn: Callable[[GridPoint, Any], Array],
        tree_grid: Grid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
    ) -> MinimumState:
        state = MinimumState(current_minimum_value=jnp.full(f_struct.shape, jnp.inf))
        return state

    def update(
        self,
        fn: Callable[[GridPoint, Any], Array],
        tree_grid_point: GridPoint,
        args: Any,
        state: MinimumState,
    ) -> MinimumState:
        raise NotImplementedError

    def postprocess(
        self,
        fn: Callable[[GridPoint, Any], Array],
        tree_grid: Grid,
        args: Any,
        state: MinimumState,
    ) -> MinimumSolution:
        return MinimumSolution(final_state=state)
