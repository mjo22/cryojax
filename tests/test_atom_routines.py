# from cryojax.simulator._scattering._gaussian_mixture import (
#    _evaluate_coord_to_grid_sq_distances,
# )
# from cryojax.simulator._scattering._gaussian_mixture import (
#    _build_pixel_grid,
#    IndependentAtomScattering,
# )

# from cryojax.simulator._density import AtomCloud

"""
class TestDistanceEvaluation:
    def test_single_point(self):
        grid1d = jnp.arange(0, 128)
        XX, YY = jnp.meshgrid(grid1d, grid1d)

        pixel_grid = jnp.expand_dims(grid1d, axis=(0, 2))
        x = jnp.array([[[0, 0, 0]]])
        sq_distance = _evaluate_coord_to_grid_sq_distances(x, pixel_grid)
        assert sq_distance.shape == (1, 128, 128, 1)

        assert jnp.allclose(sq_distance[0, :, :, 0], XX**2 + YY**2)

    def test_multiple_points(self):
        grid1d = jnp.arange(0, 128)
        XX, YY = jnp.meshgrid(grid1d, grid1d)

        pixel_grid = jnp.expand_dims(grid1d, axis=(0, 2))
        x = jnp.array([[[0, 0, 0], [0, 1, 0]]])
        sq_distance = _evaluate_coord_to_grid_sq_distances(x, pixel_grid)
        assert sq_distance.shape == (1, 128, 128, 2)

        assert jnp.allclose(sq_distance[0, :, :, 0], XX**2 + YY**2)
        assert jnp.allclose(sq_distance[0, :, :, 1], XX**2 + (YY - 1) ** 2)


class TestIndependentAtomScattering:
    @pytest.mark.parametrize("stdev_val", [1.0, 0.50])
    def test_single_atom_normalization(self, stdev_val):
        '''
        Tests that the renderer correctly normalizes the image of a single atom.
        '''
        # Set up a renderer with a single atom
        pixel_size = 0.4
        weights = jnp.array([1.0, 0.4, 0.6])
        stdevs = jnp.ones(3) * stdev_val
        variances = stdevs**2

        key = jax.random.PRNGKey(8675309)

        # Three random atoms in the center, each of the same variance
        coordinates = jax.random.uniform(key, shape=(1, 3, 3)) - 0.5
        coordinates *= 5

        # IAS = IndependentAtomScattering((100, 100))
        IAS = IndependentAtomScattering(ImageManager((50, 50)))

        ac = AtomCloud(
            weights,
            coordinates,
            variances=variances,
            identity=jnp.array([0, 1, 2]),
        )
        image = IAS.scatter(ac, pixel_size, return_Fourier=False)

        # Render the atom
        image_sum = jnp.sum(image) * pixel_size**2
        print("Image Sum:", image_sum)

        # Compute the correct normalization
        correct_norm = 2 * jnp.pi * stdevs[0] ** 2 * jnp.sum(weights)

        # Check that the image is normalized
        assert jnp.allclose(image_sum, correct_norm)
"""

# Michael: I removed the scattering method from the ElectronDensity
# representations. This is much cleaner, makes more sense, and has
# allowed for better type checking! Commenting out this test
"""
class TestAtomCloud:
    @pytest.mark.parametrize("stdev_val", [1.0, 0.50])
    def test_can_scatter(self, stdev_val):
        pixel_size = 0.4
        weights = jnp.array([1.0, 0.4, 0.6])
        variances = (jnp.ones(3) * stdev_val) ** 2
        identity = jnp.array([0, 1, 2])

        key = jax.random.PRNGKey(8675309)

        coordinates = 4 * jax.random.uniform(key, shape=(1, 3, 3)) - 2.0

        IAS = IndependentAtomScattering((100, 100))

        ac = AtomCloud(weights, coordinates, variances, identity)
        img = ac.scatter(IAS, pixel_size)

        ref_image = IAS.scatter(
            weights,
            coordinates,
            pixel_size,
            jnp.array([0, 1, 2]),
            variances,
            False,
        )
        ref_image_fft = jnp.fft.fft2(ref_image)

        assert jnp.allclose(img, ref_image_fft)

        # Three random atoms in the center, each of the same variance
"""
