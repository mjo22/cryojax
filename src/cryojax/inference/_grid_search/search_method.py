"""An interface for a grid search method."""

from abc import abstractmethod
from typing import Any, Callable, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PyTree

from .custom_types import PyTreeGrid, PyTreeGridPoint, SearchSolution, SearchState


class AbstractGridSearchMethod(
    eqx.Module, Generic[SearchState, SearchSolution], strict=True
):
    @abstractmethod
    def init(
        self, tree_grid: PyTreeGrid, f_struct: PyTree[jax.ShapeDtypeStruct]
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
    def terminate(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point: PyTreeGridPoint,
        args: Any,
        state: SearchState,
        current_iteration_index: Int[Array, ""],
        maximum_iterations: int,
    ) -> Bool[Array, ""]:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        tree_grid: PyTreeGrid,
        final_state: SearchState,
        final_iteration_index: Int[Array, ""],
        maximum_iterations: int,
    ) -> SearchSolution:
        raise NotImplementedError


class MinimumState(eqx.Module, strict=True):
    current_minimum_value: Array
    current_best_solution: Array


class MinimumSolution(eqx.Module, strict=True):
    final_state: MinimumState
    stats: dict[str, Any]

    def __init__(
        self,
        final_state: MinimumState,
        final_iteration_index: Int[Array, ""],
        maximum_iterations: int,
    ):
        final_iteration_index = eqx.error_if(
            final_iteration_index,
            final_iteration_index != jnp.array(maximum_iterations),
            "The final index of the grid search iteration was "
            "found to not be equal to the size of the grid. "
            f"The final index is {final_iteration_index}, "
            f"but it should be {maximum_iterations}.",
        )
        self.final_state = final_state
        self.stats = dict(
            final_iteration_index=final_iteration_index,
            maximum_iterations=maximum_iterations,
        )


class SearchForMinimum(
    AbstractGridSearchMethod[MinimumState, MinimumSolution], strict=True
):
    def init(
        self, tree_grid: PyTreeGrid, f_struct: PyTree[jax.ShapeDtypeStruct]
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

    def terminate(
        self,
        fn: Callable[[PyTreeGridPoint, Any], Array],
        tree_grid_point: PyTreeGridPoint,
        args: Any,
        state: MinimumState,
        current_iteration_index: Int[Array, ""],
        maximum_iterations: int,
    ) -> Bool[Array, ""]:
        return jnp.array(False)

    def postprocess(
        self,
        tree_grid: PyTreeGrid,
        final_state: MinimumState,
        final_iteration_index: Int[Array, ""],
        maximum_iterations: int,
    ) -> MinimumSolution:
        return MinimumSolution(final_state, final_iteration_index, maximum_iterations)
