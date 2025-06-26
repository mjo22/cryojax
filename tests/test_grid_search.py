import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, install_import_hook


with install_import_hook("cryojax", "typeguard.typechecked"):
    import cryojax.jax_util as cju
    from cryojax.coordinates import make_coordinate_grid


class ExampleModule(eqx.Module):
    a_1: Array
    a_2: Array
    a_3: Array
    placeholder: None

    def __init__(self, a_1, a_2, a_3):
        self.a_1 = a_1
        self.a_2 = a_2
        self.a_3 = a_3
        self.placeholder = None


def test_pytree_grid_manipulation():
    # ... make three arrays with the same leading dimension
    a_1, a_2, a_3 = tuple([jnp.arange(5) for _ in range(3)])
    # ... now two other arrays with different leading dimensions
    b, c = jnp.arange(7), jnp.arange(20)
    # Build a random tree grid
    is_leaf = lambda x: isinstance(x, ExampleModule)
    tree_grid = [ExampleModule(a_1, a_2, a_3), b, None, (c, (None,))]
    # Get grid point
    shape = cju.tree_grid_shape(tree_grid, is_leaf=is_leaf)
    tree_grid_point = cju.tree_grid_take(
        tree_grid, cju.tree_grid_unravel_index(0, tree_grid, is_leaf=is_leaf)
    )
    tree_grid_points = cju.tree_grid_take(
        tree_grid,
        cju.tree_grid_unravel_index(jnp.asarray([0, 10]), tree_grid, is_leaf=is_leaf),
    )
    # Define ground truth
    true_shape = (a_1.size, b.size, c.size)
    true_tree_grid_point = [
        ExampleModule(a_1[0], a_2[0], a_3[0]),
        b[0],
        None,
        (c[0], (None,)),
    ]
    true_tree_grid_points = [
        ExampleModule(a_1[([0, 0],)], a_2[([0, 0],)], a_3[([0, 0],)]),
        b[([0, 0],)],
        None,
        (c[([0, 10],)], (None,)),
    ]
    assert shape == true_shape
    assert eqx.tree_equal(tree_grid_point, true_tree_grid_point)
    assert eqx.tree_equal(tree_grid_points, true_tree_grid_points)


@eqx.filter_jit
def cost_fn(grid_point, variance_plus_offset):
    variance, offset = variance_plus_offset
    mu_x, mu_y = offset
    x, y = grid_point
    return -jnp.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * variance)) / jnp.sqrt(
        2 * jnp.pi * variance
    )


@pytest.mark.parametrize(
    "batch_size,dim,offset,variance",
    [
        (None, 200, (-1.0, 2.0), 10.0),
        (1, 200, (-1.0, 2.0), 10.0),
        (10, 200, (-1.0, 2.0), 10.0),
        (33, 200, (99.0, 99.0), 10.0),
    ],
)
def test_run_grid_search(batch_size, dim, offset, variance):
    # Compute full landscape of simple analytic "cost function"
    coords = make_coordinate_grid((dim, dim))
    variance, offset = jnp.asarray(variance), jnp.asarray(offset)
    landscape = jax.vmap(jax.vmap(cost_fn, in_axes=[0, None]), in_axes=[0, None])(
        coords, (variance, offset)
    )
    # Find the true minimum value and its location
    true_min_eval = landscape.min()
    true_min_idx = jnp.squeeze(jnp.argwhere(landscape == true_min_eval))
    true_min_pos = tuple(coords[true_min_idx[0], true_min_idx[1]])
    # Generate a sparse representation of coordinate grid
    x, y = (
        jnp.fft.fftshift(jnp.fft.fftfreq(dim)) * dim,
        jnp.fft.fftshift(jnp.fft.fftfreq(dim)) * dim,
    )
    grid = (x, y)
    # Run the grid search
    method = cju.MinimumSearchMethod(batch_size=batch_size)
    solution = cju.run_grid_search(cost_fn, method, grid, (variance, offset))
    np.testing.assert_allclose(solution.state.current_minimum_eval, true_min_eval)
    np.testing.assert_allclose(solution.value, true_min_pos)
