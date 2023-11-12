from jax import numpy as jnp
from cryojax.simulator.scattering import _evaluate_coord_to_grid_sq_distances

class TestDistanceEvaluation():
    def test_single_point(self):
        grid1d = jnp.arange(0, 128)
        XX, YY = jnp.meshgrid(grid1d, grid1d)

        pixel_grid  = jnp.expand_dims(grid1d, axis=(0, 2))
        x = jnp.array([[[0, 0, 0]]])
        sq_distance = _evaluate_coord_to_grid_sq_distances(x, pixel_grid)
        assert sq_distance.shape == (1, 128, 128, 1)

        assert jnp.allclose(sq_distance[0, :, :, 0], XX**2 + YY**2)

    def test_multiple_points(self):
        grid1d = jnp.arange(0, 128)
        XX, YY = jnp.meshgrid(grid1d, grid1d)

        pixel_grid  = jnp.expand_dims(grid1d, axis=(0, 2))
        x = jnp.array([[[0, 0, 0], [0, 1, 0]]])
        sq_distance = _evaluate_coord_to_grid_sq_distances(x, pixel_grid)
        assert sq_distance.shape == (1, 128, 128, 2)

        assert jnp.allclose(sq_distance[0, :, :, 0], XX**2 + YY**2)
        assert jnp.allclose(sq_distance[0, :, :, 1], XX**2 + (YY-1)**2)
