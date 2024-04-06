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
    """An abstract interface that determines the behavior of the grid
    search.
    """

    @abstractmethod
    def init(
        self,
        tree_grid: PyTreeGrid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> SearchState:
        """Initialize the state of the search method.

        **Arguments:**

        - `tree_grid`: As [`run_grid_search`][].
        - `f_struct`: A container that stores the `shape` and `dtype`
                      returned by `fn`.
        - `is_leaf`: As [`run_grid_search`][].

        **Returns:**

        Any pytree that represents the state of the grid search.
        """
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
        """Update the state of the grid search.

        **Arguments:**

        - `fn`: As [`run_grid_search`][].
        - `tree_grid_point`: The grid point at which to evaluate `fn`. Specifically,
                             `fn` is evaluated as `fn(tree_grid_point, args)`.
        - `args`: As [`run_grid_search`][].
        - `state`: The current state of the search.
        - `current_iteration_index`: The current index of the search. This is
                                     used to index `tree_grid` to extract the
                                     `tree_grid_point`.

        **Returns:**

        The updated state of the grid search.
        """
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
        """Post-process the final state of the grid search into a
        solution.

        **Arguments:**

        - `tree_grid`: As [`run_grid_search`][].
        - `final_state`: The final state of the grid search.
        - `f_struct`: A container that stores the `shape` and `dtype`
                      returned by `fn`.
        - `is_leaf`: As [`run_grid_search`][].

        **Returns:**

        Any pytree that represents the solution of the grid search.
        """
        raise NotImplementedError


class MinimumState(eqx.Module, strict=True):
    current_minimum_eval: Array
    current_best_raveled_index: Array


class MinimumSolution(eqx.Module, strict=True):
    value: PyTreeGridPoint
    stats: dict[str, Any]
    state: MinimumState


class SearchForMinimum(
    AbstractGridSearchMethod[MinimumState, MinimumSolution], strict=True
):
    """Simply find the minimum value returned by `fn` over all grid points.

    The minimization is done *elementwise* for the output returned by `fn(y, args)`.
    """

    def init(
        self,
        tree_grid: PyTreeGrid,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> MinimumState:
        # Initialize the state, just keeping track of the best function values
        # and their respective grid index
        state = MinimumState(
            current_minimum_eval=jnp.full(f_struct.shape, jnp.inf),
            current_best_raveled_index=jnp.full(f_struct.shape, 0, dtype=int),
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
        # Unpack the current state
        last_minimum_value = state.current_minimum_eval
        last_best_raveled_index = state.current_best_raveled_index
        # Evaluate the function
        value = fn(tree_grid_point, args)
        # Update the minimum and best grid index, elementwise
        is_less_than_last_minimum = value < last_minimum_value
        current_minimum_eval = jnp.where(
            is_less_than_last_minimum, value, last_minimum_value
        )
        current_best_raveled_index = jnp.where(
            is_less_than_last_minimum, current_iteration_index, last_best_raveled_index
        )
        return MinimumState(current_minimum_eval, current_best_raveled_index)

    def postprocess(
        self,
        tree_grid: PyTreeGrid,
        final_state: MinimumState,
        f_struct: PyTree[jax.ShapeDtypeStruct],
        *,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> MinimumSolution:
        # Make sure that shapes did not get modified during loop
        if final_state.current_best_raveled_index.shape != f_struct.shape:
            raise ValueError(
                "The shape of the search state solution does "
                "not match the shape of the output of `fn`. Got "
                f"output shape {f_struct.shape} for `fn`, but got "
                f"shape {final_state.current_best_raveled_index.shape} for the "
                "solution."
            )
        # Extract the solution of the search, i.e. the grid point(s) corresponding
        # to the raveled grid index
        if f_struct.shape == ():
            raveled_index = final_state.current_best_raveled_index
        else:
            raveled_index = final_state.current_best_raveled_index.ravel()
        # ... get the pytree representation of the index
        tree_grid_index = tree_grid_unravel_index(
            raveled_index, tree_grid, is_leaf=is_leaf
        )
        # ... index the full grid, reshaping the solution's leaves to be the same
        # shape as returned by `fn`
        value = jtu.tree_map(
            lambda x: x.reshape(f_struct.shape),
            tree_grid_take(tree_grid, tree_grid_index),
        )
        # ... build and return the solution
        return MinimumSolution(
            value,
            {"n_iterations": math.prod(tree_grid_shape(tree_grid, is_leaf=is_leaf))},
            final_state,
        )
