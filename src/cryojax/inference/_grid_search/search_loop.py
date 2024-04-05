"""The main search loop for the grid search."""

import math
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Int, PyTree

from .custom_types import PyTreeGrid, PyTreeGridIndex, PyTreeGridPoint
from .search_method import AbstractGridSearchMethod


@eqx.filter_jit
def run_grid_search(
    fn: Callable[[PyTreeGridPoint, Any], Array],
    method: AbstractGridSearchMethod,
    tree_grid: PyTreeGrid,
    args: Any = None,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTree[Any]:
    """Run a grid search to optimize the function `fn`.

    **Arguments:**

    - `fn`: The function we would like to minimize with grid search. This
            should be evaluated at arguments `fn(y, args)` and can return
            any `jax.Array`. Here, `y` is a particular grid point of
            `tree_grid`.
    - `method`: An interface that specifies what we would like to do with
                each evaluation of `fn`.
    - `tree_grid`:
    - `args`: Arguments passed to `fn`.
    - `is_leaf`: As [`jax.tree_util.tree_flatten`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_flatten.html).
                 This specifies what is to be treated as a leaf in `tree_grid`.

    **Returns:**

    Any pytree, as specified by the method `AbstractGridSearchMethod.postprocess`.
    """
    # Evaluate the shape and dtype of the output of `fn` using
    # eqx.filter_closure_convert
    initial_iteration_index = jnp.array(0)
    init_tree_grid_point = tree_grid_take(
        tree_grid,
        tree_grid_unravel_index(initial_iteration_index, tree_grid, is_leaf=is_leaf),
    )
    fn = eqx.filter_closure_convert(fn, init_tree_grid_point, args)
    f_struct = jtu.tree_map(
        lambda x: x.value,
        jtu.tree_map(eqxi.Static, fn.out_struct),
        is_leaf=lambda x: isinstance(x, eqxi.Static),
    )
    # Get the initial state of the search method
    init_state = method.init(tree_grid, f_struct)
    dynamic_init_state, static_state = eqx.partition(init_state, eqx.is_array)
    # Get the number of iterations of the loop (the size of the grid)
    maximum_iterations = math.prod(tree_grid_shape(tree_grid, is_leaf=is_leaf))
    # Finally, build the loop
    init_carry = (
        init_tree_grid_point,
        initial_iteration_index,
        dynamic_init_state,
        tree_grid,
    )

    def cond_fun(carry):
        tree_grid_point, iteration_index, dynamic_state, _ = carry
        state = eqx.combine(static_state, dynamic_state)
        terminate = method.terminate(
            fn, tree_grid_point, args, state, iteration_index, maximum_iterations
        )
        return jnp.invert(terminate)

    def body_fun(carry):
        tree_grid_point, iteration_index, dynamic_state, tree_grid = carry
        state = eqx.combine(static_state, dynamic_state)
        new_state = method.update(fn, tree_grid_point, args, state, iteration_index)
        new_dynamic_state, new_static_state = eqx.partition(new_state, eqx.is_array)
        assert eqx.tree_equal(static_state, new_static_state) is True
        new_iteration_index = iteration_index + 1
        new_tree_grid_point = tree_grid_take(
            tree_grid,
            tree_grid_unravel_index(new_iteration_index, tree_grid, is_leaf=is_leaf),
        )
        return new_tree_grid_point, new_iteration_index, new_dynamic_state, tree_grid

    # Run and unpack results
    final_carry = eqxi.while_loop(
        cond_fun, body_fun, init_carry, kind="lax", max_steps=maximum_iterations
    )
    _, final_iteration_index, dynamic_final_state, _ = final_carry
    final_state = eqx.combine(static_state, dynamic_final_state)
    # Return the solution
    solution = method.postprocess(
        tree_grid, final_state, final_iteration_index, maximum_iterations
    )
    return solution


def tree_grid_shape(
    tree_grid: PyTreeGrid,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> tuple[int, ...]:
    """Get the shape of a pytree grid.

    **Arguments:**

    - `tree_grid`: A sparse grid cartesian grid, represented as a pytree.
                   See [`run_grid_search`][] for more information.
    - `is_leaf`: As [`jax.tree_util.tree_flatten`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_flatten.html).

    **Returns:**

    The shape of `tree_grid`.

    !!! Example

        ```python
        # A simple "pytree grid"
        simple_tree_grid = (jnp.zeros(10), jnp.zeros(10), jnp.zeros(10))
        # Its shape is just the shape of the cartesian product of its leaves
        assert tree_grid_shape(simple_tree_grid) == (10, 10, 10)
        ```

    !!! Example

        ```python
        # Library code
        import equinox as eqx
        import jax

        class SomeModule(eqx.Module):

            a: jax.Array

        # End-user script
        # ... create a more complicated grid
        complicated_tree_grid = (SomeModule(jnp.zeros(10)), jnp.zeros(10), (jnp.zeros(10), None))
        # Its shape is still just the shape of the cartesian product of its leaves
        assert tree_grid_shape(complicated_tree_grid) == (10, 10, 10)
        ```
    """  # noqa: E501
    _leading_dim_resolver = jtu.tree_map(
        _LeafLeadingDimension, tree_grid, is_leaf=is_leaf
    )
    _reduce_fn = lambda x, y: (
        x.get() + y.get() if isinstance(x, _LeafLeadingDimension) else x + y.get()
    )
    return jtu.tree_reduce(
        _reduce_fn,
        _leading_dim_resolver,
        is_leaf=lambda x: isinstance(x, _LeafLeadingDimension),
    )


def tree_grid_unravel_index(
    raveled_index: int | Int[Array, ""] | Int[Array, " _"],
    tree_grid: PyTreeGrid,
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> PyTreeGridIndex:
    """Get a "grid index" for a pytree grid.

    Roughly, this can be thought of as `jax.numpy.unravel_index`, but with a
    pytree grid. See [`tree_grid_take`][] for an example of how to use this
    function to sample a grid point.

    **Arguments:**

    - `raveled_index`: A flattened index for `tree_grid`. Simply pass an integer
                       valued index, as one would with a flattened array. Passing
                       a 1D array of indices is also supported.
    - `tree_grid`: A sparse grid cartesian grid, represented as a pytree.
                   See [`run_grid_search`][] for more information.
    - `is_leaf`: As [`jax.tree_util.tree_flatten`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_flatten.html).

    **Returns:**

    The grid index. This is a pytree of the same structure as `tree_grid`, with the
    result of `jax.numpy.unravel_index(raveled_index, shape)` inserted into the
    appropriate leaf. Here, `shape` is given by the output of [`tree_grid_shape`][].
    """
    raveled_index = jnp.asarray(raveled_index)
    shape = tree_grid_shape(tree_grid, is_leaf=is_leaf)
    # raveled_index = eqx.error_if(
    #     raveled_index,
    #     jnp.logical_or(raveled_index < 0, raveled_index >= math.prod(shape)),
    #     "The flattened grid index must be greater than 0 and less than the "
    #     f"grid size. Got index {raveled_index}, but the grid has shape {shape}, "
    #     f"so its maximum index is {math.prod(shape) - 1}.",
    # )
    unraveled_index = jnp.unravel_index(raveled_index, shape)
    tree_grid_def = jtu.tree_structure(tree_grid, is_leaf=is_leaf)
    tree_grid_index = jtu.tree_unflatten(tree_grid_def, unraveled_index)

    return tree_grid_index


def tree_grid_take(
    tree_grid: PyTreeGrid,
    tree_grid_index: PyTreeGridIndex,
) -> PyTreeGridPoint:
    """Get a grid point of the pytree grid, given a
    pytree grid index. See [`tree_grid_unravel_index`][] to see
    how to return a pytree grid index.

    Roughly, this can be thought of as `jax.numpy.take`, but with a
    pytree grid.

    **Arguments:**

    - `tree_grid`: A sparse cartesian grid, represented as a pytree.
                   See [`run_grid_search`][] for more information.
    - `tree_grid_index`: An index for `tree_grid`, also represented as a pytree.
                   See [`tree_grid_unravel_index`][] for more information.

    **Returns:**

    A grid point of a pytree grid. This is a pytree of the same
    structure as `tree_grid` (or a prefix of it), where each leaf
    is indexed by the leaf at `tree_grid_index`.

    !!! Example

        ```python
        # A simple "pytree grid"
        simple_tree_grid = (jnp.zeros(10), jnp.zeros(10), jnp.zeros(10))
        # Its shape is just the shape of the cartesian product of its leaves
        raveled_index = 7
        tree_grid_index = tree_grid_unravel_index(raveled_index, simple_tree_grid)
        tree_grid_point = tree_grid_take(simple_tree_grid, tree_grid_index)
        assert tree_grid_point == (jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0.))
        ```
    """
    tree_grid_point = _tree_take(tree_grid, tree_grid_index, axis=0)
    return tree_grid_point


def _tree_take(
    pytree_of_arrays: PyTree[Array],
    pytree_of_indices: PyTree[Int[Array, "..."]],
    axis: Optional[int] = None,
    mode: Optional[str] = None,
    fill_value: Optional[Array] = None,
) -> PyTree[Array]:
    return jtu.tree_map(
        lambda i, l: _leaf_take(i, l, axis=axis, mode=mode, fill_value=fill_value),
        pytree_of_indices,
        pytree_of_arrays,
    )


def _leaf_take(index, leaf, **kwargs):
    _take_fn = lambda array: jnp.take(jnp.atleast_1d(array), index, **kwargs)
    if eqx.is_array(leaf):
        return _take_fn(leaf)
    else:
        return jtu.tree_map(_take_fn, leaf)


def _get_leading_dim(array):
    return (array.shape[0],)


class _LeafLeadingDimension(eqx.Module):
    _leaf: Any

    def get(self):
        if eqx.is_array(self._leaf):
            return _get_leading_dim(self._leaf)
        else:
            leaves = jtu.tree_leaves(self._leaf)
            if len(leaves) > 0:
                _leading_dim = _get_leading_dim(leaves[0])
                if not all([_get_leading_dim(leaf) == _leading_dim for leaf in leaves]):
                    raise ValueError(
                        "Arrays stored in PyTree leaves should share the same "
                        "leading dimension. Found that this is not true for "
                        f"leaf {self._leaf}."
                    )
                return _leading_dim
            else:
                raise ValueError(f"No arrays found at leaf {self._leaf}")
