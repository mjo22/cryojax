"""An interface for a grid search method."""

import math
from abc import abstractmethod
from typing import Any, Callable, Generic, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Int, PyTree

from .custom_types import PyTreeGrid, PyTreeGridPoint, SearchSolution, SearchState
from .pytree_manipulation import (
    tree_grid_shape,
    tree_grid_take,
    tree_grid_unravel_index,
)


class AbstractGridSearchMethod(
    eqx.Module, Generic[SearchState, SearchSolution], strict=True
):
    @abstractmethod
    def init(
        self,
        tree_grid: PyTreeGrid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> SearchState:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point: PyTreeGridPoint,
        args: Any,
        state: SearchState,
        current_iteration_index: Int[Array, ""],
    ) -> SearchState:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        tree_grid: PyTreeGrid,
        final_state: SearchState,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> SearchSolution:
        raise NotImplementedError


class MinimumState(eqx.Module, strict=True):
    current_minimum_value: Array
    current_best_solution: Array


class MinimumSolution(eqx.Module, strict=True):
    value: PyTreeGridPoint
    stats: dict[str, Any]
    state: MinimumState


class SearchForMinimum(
    AbstractGridSearchMethod[MinimumState, MinimumSolution], strict=True
):
    def init(
        self,
        tree_grid: PyTreeGrid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> MinimumState:
        state = MinimumState(
            current_minimum_value=jnp.full(f_struct.shape, jnp.inf),
            current_best_solution=jnp.full(f_struct.shape, 0, dtype=int),
        )
        return state

    def update(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point: PyTreeGridPoint,
        args: Any,
        state: MinimumState,
        current_iteration_index: Int[Array, ""],
    ) -> MinimumState:
        value = fn(tree_grid_point, args)
        last_minimum_value = state.current_minimum_value
        last_best_solution = state.current_best_solution
        is_less_than_last_minimum = value < last_minimum_value
        current_minimum_value = jnp.where(
            is_less_than_last_minimum, value, last_minimum_value
        )
        current_best_solution = jnp.where(
            is_less_than_last_minimum, current_iteration_index, last_best_solution
        )
        return MinimumState(current_minimum_value, current_best_solution)

    def postprocess(
        self,
        tree_grid: PyTreeGrid,
        final_state: MinimumState,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> MinimumSolution:
        if final_state.current_best_solution.shape != f_struct.shape:
            raise ValueError(
                "The shape of the search state solution does "
                "not match the shape of the output of `fn`. Got "
                f"output shape {f_struct.shape} for `fn`, but got "
                f"shape {final_state.current_best_solution.shape} for the "
                "solution."
            )
        if f_struct.shape == ():
            raveled_index = final_state.current_best_solution
        else:
            raveled_index = final_state.current_best_solution.ravel()
        tree_grid_index = tree_grid_unravel_index(
            raveled_index, tree_grid, is_leaf=is_leaf
        )
        value = jtu.tree_map(
            lambda x: x.reshape(f_struct.shape),
            tree_grid_take(tree_grid, tree_grid_index),
        )
        return MinimumSolution(
            value,
            {"n_iterations": math.prod(tree_grid_shape(tree_grid, is_leaf=is_leaf))},
            final_state,
        )
