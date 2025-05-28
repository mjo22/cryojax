import pathlib
from typing import Any, Callable, Optional, TypeVar

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, Int

from .base_particle_dataset import (
    AbstractParticleParameterDataset,
    AbstractParticleParameters,
    AbstractParticleStackDataset,
)


PerParticleT = TypeVar("PerParticleT")
ConstantT = TypeVar("ConstantT")
T = TypeVar("T")


def write_simulated_image_stack(
    dataset: AbstractParticleStackDataset,
    compute_image_fn: Callable[
        [AbstractParticleParameters, ConstantT, PerParticleT],
        Float[Array, "_ _"],
    ],
    constant_args: ConstantT = None,
    per_particle_args: PerParticleT = None,
    batch_size: Optional[int] = None,
    path_to_output: Optional[str | pathlib.Path] = None,
    **kwargs: Any,
):
    """Write a stack of images from parameters contained in an
    `AbstractParticleStackDataset`.

    !!! note
        This function works generally for a `compute_image_fn`
        of the form

        ```python
        image = compute_image_fn(
            parameters, constant_args, per_particle_args
        )
        ```

        where `parameters` is a cryojax `AbstractParticleParameters`
        pytree, `constant_args` is a parameter that does not change
        between images, and `per_particle_args` is a pytree whose
        leaves have a batch dimension equal to the number of particles
        to be simulated.

    *Example 1*: Basic usage such as instantiating an
    `AbstractParticleStackDataset` and writing a
    `compute_image_fn`

    ```python
    import cryojax.simulator as cxs
    import jax
    from cryojax.data import RelionParticleStackDataset
    from jaxtyping import PyTree

    # Load a `RelionParticleStackDataset` object. This loads
    # parameters and writes images
    dataset = RelionParticleStackDataset(..., mode='w')

    # Write your `compute_image_fn` function, building a
    # `ContrastImageModel` (see tutorials for details)

    def compute_image_fn(
        parameters: RelionParticleParameters,
        constant_args: PyTree,
        _,
    ) -> jax.Array:
        # `constant_args` do not change between images. For
        # example, include the method of taking projections
        potential_integrator, ... = constant_args
        # Using the pose, CTF, and config from the
        # `parameters`, build image simulation model and
        # compute
        image_model = cxs.ContrastImageModel(...)

        return image_model.render()

    # Simulate images and write to disk
    write_simulated_image_stack(
        dataset,
        compute_image_fn,
        constant_args=(potential_integrator, ...)
        per_particle_args=None, # default
        batch_size=10,
    )
    ```

    *Example 2*: More-advanced usage, writing a
    `compute_image_fn` that simulates images with noise.
    Uses `per_particle_args` as well as `constant_args`.

    ```python
    import cryojax.simulator as cxs
    import jax
    from cryojax.data import RelionParticleStackDataset
    from jaxtyping import Array, PyTree, Shaped

    # Load a `RelionParticleStackDataset` object. This loads
    # parameters and writes images
    dataset = RelionParticleStackDataset(..., mode='w')

    # Instantiate per-particle arguments. First, the RNG keys used
    # to generate the noise
    seed = 0
    key = jax.random.key(seed)
    key, *keys_noise = jax.random.split(key, n_images+1)
    keys_noise = jnp.array(keys_noise)
    # ... then, add a scaling parameter for the images
    key, subkey = jax.random.split(key)
    scaling_params = jax.random.uniform(subkey, shape=(n_images,))

    # Now write your `compute_image_fn` function, building a
    # `cryojax.distributions.IndependentGaussianPixels` to
    # simulate images with white noise (see tutorials for details)

    def compute_image_fn(
        particle_parameters: RelionParticleParameters,
        constant_args: PyTree,
        per_particle_args: PyTree[Shaped[Array, "_ ..."]],
    ) -> jax.Array:
        ... = constant_args
        key, scale = per_particle_args

        # Combine two previously split PyTrees
        image_model = cxs.ContrastImageModel(...)
        distribution = cxs.IndependentGaussianPixels(image_model, ...)

        return scale * distribution.sample(key)

    write_simulated_image_stack(
        dataset,
        compute_image_fn,
        constant_args=(...)
        per_particle_args=(keys_noise, scaling_params)
        batch_size=10,
    )
    ```

    **Arguments:**

    - `dataset`:
        The `AbstractParticleStackDataset` dataset. Note that this must be
        passed in *writing mode*, i.e. `mode = 'w'`.
    - `compute_image_fn`:
        A callable that computes the image stack from the parameters contained
        in the STAR file.
    - `constant_args`:
        The constant arguments to pass to the `compute_image_fn` function.
        These must be the same for all images.
    - `per_particle_args`:
        Arguments to pass to the `compute_image_fn` function.
        This is a pytree with leaves having a batch size with equal dimension
        to the number of images.
    - `batch_size`:
        The number of particle indices to compute in parallel and
        store in a single image file. If an integer, compute `batch_size`
        images at a time using `equinox.filter_vmap`. If `None`, simulate
        images in a python for-loop.
    -  `path_to_output`:
        The location to write the metadata file, i.e. the
        `AbstractParticleStackDataset.parameter_dataset.path_to_output`.
    - `kwargs`:
        Keyword arguments passed to
        `AbstractParticleStackDataset.parameter_dataset.save`.
    """
    if dataset.mode == "r":
        raise ValueError(
            "Found that the `dataset` was in reading mode "
            "(`mode = 'r'`), but this must be instantiated in "
            "writing mode (`mode = 'w'`)."
        )
    n_particles = len(dataset)
    if batch_size is None:
        do_vmap, batch_size = False, 1
    else:
        do_vmap = True

    # Get function that simulates batch of images
    compute_image_stack_fn = _configure_simulation_fn(
        compute_image_fn, batch_size, do_vmap
    )
    # Run control flow
    n_batches, remainder = n_particles // batch_size, n_particles % batch_size
    parameter_dataset = dataset.parameter_dataset
    for batch_index in range(n_batches):
        index_array = np.arange(
            batch_index * batch_size, (batch_index + 1) * batch_size, dtype=int
        )
        images, parameters = _simulate_images(
            index_array,
            parameter_dataset,
            compute_image_stack_fn,
            constant_args,
            per_particle_args,
        )
        dataset.write_images(index_array, images, parameters)
    # ... handle remainder
    if remainder > 0:
        compute_image_stack_fn = _configure_simulation_fn(
            compute_image_fn, remainder, do_vmap
        )
        index_array = np.arange(n_particles - remainder, n_particles, dtype=int)
        images, parameters = _simulate_images(
            index_array,
            parameter_dataset,
            compute_image_stack_fn,
            constant_args,
            per_particle_args,
        )
        dataset.write_images(index_array, images, parameters)
    # Finally, save metadata file
    if path_to_output is not None:
        parameter_dataset.path_to_output = path_to_output
    parameter_dataset.save(**kwargs)


def _simulate_images(
    index: Int[np.ndarray, " _"],
    parameter_dataset: AbstractParticleParameterDataset,
    compute_image_stack_fn: Callable[
        [AbstractParticleParameters, ConstantT, PerParticleT],
        Float[Array, "_ _ _"],
    ],
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
) -> tuple[Float[Array, "_ _ _"], AbstractParticleParameterDataset]:
    parameters = parameter_dataset[index]
    args = (constant_args, _index_pytree(index, per_particle_args))
    image_stack = compute_image_stack_fn(parameters, *args)

    return image_stack, parameters


def _configure_simulation_fn(
    compute_image_fn: Callable[
        [AbstractParticleParameters, ConstantT, PerParticleT],
        Float[Array, "_ _"],
    ],
    batch_size: int,
    do_vmap: bool,
) -> Callable[
    [AbstractParticleParameters, ConstantT, PerParticleT],
    Float[Array, "_ _ _"],
]:
    if do_vmap:
        compute_image_stack_fn = eqx.filter_jit(
            eqx.filter_vmap(
                compute_image_fn, in_axes=(eqx.if_array(0), None, eqx.if_array(0))
            )
        )
    else:

        def compute_image_stack_fn(parameters, constant_args, per_particle_args):
            shape = parameters.instrument_config.shape
            image_stack = np.empty((batch_size, *shape))
            for i in range(batch_size):
                parameters_at_i = _index_pytree(i, parameters)
                per_particle_args_at_i = _index_pytree(i, per_particle_args)
                image = compute_image_fn(
                    parameters_at_i, constant_args, per_particle_args_at_i
                )
                image_stack[i] = np.asarray(image)
            return image_stack

    return compute_image_stack_fn  # type: ignore


def _index_pytree(
    index: int | Int[np.ndarray, ""] | Int[np.ndarray, " _"], pytree: T
) -> T:
    dynamic, static = eqx.partition(pytree, eqx.is_array)
    dynamic_at_index = jax.tree.map(lambda x: x[index], dynamic)
    return eqx.combine(dynamic_at_index, static)
