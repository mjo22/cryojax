import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from cryojax.inference import tree_grid_shape, tree_grid_take, tree_grid_unravel_index


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


def test_pytree_grid_shape():
    # ... make three arrays with the same leading dimension
    a_1, a_2, a_3 = tuple([jnp.arange(5) for _ in range(3)])
    # ... now two other arrays with different leading dimensions
    b, c = jnp.arange(7), jnp.arange(20)
    # Build a random tree grid
    is_leaf = lambda x: isinstance(x, ExampleModule)
    tree_grid = [ExampleModule(a_1, a_2, a_3), b, None, (c, (None,))]
    # Get grid point
    shape = tree_grid_shape(tree_grid, is_leaf=is_leaf)
    tree_grid_point = tree_grid_take(
        tree_grid, tree_grid_unravel_index(0, tree_grid, is_leaf=is_leaf)
    )
    tree_grid_points = tree_grid_take(
        tree_grid,
        tree_grid_unravel_index(jnp.asarray([0, 10]), tree_grid, is_leaf=is_leaf),
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
