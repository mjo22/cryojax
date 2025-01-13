import os
import pathlib
from typing import Any, Callable, cast, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import starfile
from jaxtyping import Array, Float, PRNGKeyArray

from ...image.operators import Constant, FourierGaussian
from ...io import write_image_stack_to_mrc
from ...utils import get_filter_spec
from ._starfile_reading import RelionParticleMetadata, RelionParticleParameters


def _get_filename(step, n_char=6):
    if step == 0:
        fname = "0" * n_char
    else:
        n_dec = int(np.log10(step))
        fname = "0" * (n_char - n_dec) + str(step)
    return fname


def write_starfile_with_particle_parameters(
    particle_parameters: RelionParticleParameters,
    filename: str | pathlib.Path,
    mrc_batch_size: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Generate a STAR file from a RelionParticleStack object.

    This function does not generate particles, it merely populates the starfile.

    The starfile is written to disc at the location specified by filename.

    **Arguments:**

    - `particle_parameters`:
        The `RelionParticleParameters` object.
    - `filename`:
        The filename of the STAR file to write.
    - `mrc_batch_size`:
        The number of images to write to each MRC file. If `None`, the number of
        images in the `RelionParticleStack` is used.
    - `overwrite`:
        Whether to overwrite the STAR file if it already exists.
    """

    path_to_starfile = os.path.dirname(filename)
    if not os.path.exists(path_to_starfile):
        os.makedirs(path_to_starfile)

    if not overwrite and os.path.exists(filename):
        raise FileExistsError(
            f"Overwrite was set to False, but STAR file {filename} already exists."
        )

    n_images = particle_parameters.pose.offset_x_in_angstroms.shape[0]

    if mrc_batch_size is None:
        mrc_batch_size = n_images

    assert (
        n_images >= mrc_batch_size
    ), "n_images must be greater than or equal to mrc_batch_size"

    starfile_dict = dict()
    # Generate optics group
    optics_df = pd.DataFrame()
    optics_df["rlnOpticsGroup"] = [1]
    optics_df["rlnVoltage"] = particle_parameters.instrument_config.voltage_in_kilovolts
    optics_df["rlnSphericalAberration"] = (
        particle_parameters.transfer_theory.ctf.spherical_aberration_in_mm
    )
    optics_df["rlnImagePixelSize"] = particle_parameters.instrument_config.pixel_size
    optics_df["rlnImageSize"] = particle_parameters.instrument_config.shape[0]
    optics_df["rlnAmplitudeContrast"] = (
        particle_parameters.transfer_theory.ctf.amplitude_contrast_ratio
    )
    starfile_dict["optics"] = optics_df

    # Generate particles group
    particles_df = pd.DataFrame()

    # fixed values
    particles_df["rlnCtfMaxResolution"] = np.zeros(n_images)
    particles_df["rlnCtfFigureOfMerit"] = np.zeros(n_images)
    particles_df["rlnClassNumber"] = np.ones(n_images)
    particles_df["rlnOpticsGroup"] = np.ones(n_images)

    particles_df["rlnOriginXAngst"] = particle_parameters.pose.offset_x_in_angstroms
    particles_df["rlnOriginYAngst"] = particle_parameters.pose.offset_y_in_angstroms
    particles_df["rlnAngleRot"] = particle_parameters.pose.view_phi
    particles_df["rlnAngleTilt"] = particle_parameters.pose.view_theta
    particles_df["rlnAnglePsi"] = particle_parameters.pose.view_psi
    particles_df["rlnDefocusU"] = (
        particle_parameters.transfer_theory.ctf.defocus_in_angstroms
        + particle_parameters.transfer_theory.ctf.astigmatism_in_angstroms / 2
    )
    particles_df["rlnDefocusV"] = (
        particle_parameters.transfer_theory.ctf.defocus_in_angstroms
        - particle_parameters.transfer_theory.ctf.astigmatism_in_angstroms / 2
    )
    particles_df["rlnDefocusAngle"] = (
        particle_parameters.transfer_theory.ctf.astigmatism_angle
    )

    if isinstance(particle_parameters.transfer_theory.envelope, FourierGaussian):
        particles_df["rlnCtfBfactor"] = (
            particle_parameters.transfer_theory.envelope.b_factor
        )
        particles_df["rlnCtfScalefactor"] = (
            particle_parameters.transfer_theory.envelope.amplitude
        )

    elif isinstance(particle_parameters.transfer_theory.envelope, Constant):
        particles_df["rlnCtfBfactor"] = 0.0
        particles_df["rlnCtfScalefactor"] = (
            particle_parameters.transfer_theory.envelope.value
        )

    else:
        raise NotImplementedError(
            "Only FourierGaussian and Constant envelopes are supported"
        )

    particles_df["rlnPhaseShift"] = particle_parameters.transfer_theory.ctf.phase_shift

    n_batches = n_images // mrc_batch_size
    n_remainder = n_images % mrc_batch_size

    image_names = []

    for step in range(n_batches):
        mrc_filename = _get_filename(step, n_char=6)
        mrc_relative_path = mrc_filename + ".mrcs"
        image_names += [
            _get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(mrc_batch_size)
        ]

    if n_remainder > 0:
        mrc_filename = _get_filename(n_batches, n_char=6)
        mrc_relative_path = mrc_filename + ".mrcs"
        image_names += [
            _get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(n_remainder)
        ]

    particles_df["rlnImageName"] = image_names
    starfile_dict["particles"] = particles_df
    starfile.write(starfile_dict, pathlib.Path(filename))

    return


def write_simulated_image_stack_from_starfile(
    metadata: RelionParticleMetadata,
    compute_image: (
        Callable[[RelionParticleParameters, Any], Float[Array, "y_dim x_dim"]]
        | Callable[
            [PRNGKeyArray, RelionParticleParameters, Any],
            Float[Array, "y_dim x_dim"],
        ]
    ),
    args: Any,
    is_jittable: bool = True,
    batch_size_per_mrc: Optional[int] = None,
    seed: Optional[int] = None,  # seed for the noise
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    """Writes a stack of images from the parameters contained in a
    STAR file.

    !!! note
        This function works generally for a function that computes
        images of the form `compute_image_stack(pytree, args)` or
        `compute_image_stack(key, pytree, args)`, where `key` is a
        random number generator key(s), and `pytree` and `args` are
        the two pytrees resulting from the `eqx.partition` function,
        using the `vmap_filter_spec` as the filter.

    ```python

    # Example 1: Using the function with a `compute_image_stack`
    # function that does not take a key

    # start from a previously defined `imaging_pipeline`
    # (see our Tutorials for details on how to do this)
    imaging_pipeline = ContrastImagingPipeline(...)

    # and a `RelionParticleMetadata` object
    metadata = RelionParticleMetadata(...)

    # Write your `compute_image_stack` function.

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=( 0, None), out_axes=0)
    def compute_image_stack(
        img_pipeline_vmap: AbstractImagingPipeline,
        img_pipeline_novmap: AbstractImagingPipeline,
    ):

        # Combine two previously split PyTrees
        img_pipeline = eqx.combine(img_pipeline_vmap, img_pipeline_novmap)

        return img_pipeline.render()

    write_simulated_image_stack_from_starfile(
        metadata,
        compute_image_stack,
        imaging_pipeline,
        seed=None, # our image pipeline does not require a seed
        overwrite=True,
    )

    ```

    ```python
    # Example 2: Using the function with a `compute_image_stack`
    # function that takes a key

    # start from a previously defined cryojax `distribution`
    # (see our Tutorials for details on how to do this)

    distribution = cryojax.inference.IndependentGaussianFourierModes(...)

    # and a `RelionParticleMetadata` object
    metadata = RelionParticleMetadata(...)

    # Write your `compute_image_stack` function

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(0, 0, None), out_axes=0)
    def compute_noisy_image_stack(
        key: PRNGKeyArray,
        dist_vmap: dist.AbstractDistribution,
        dist_novmap: dist.AbstractDistribution,
    ):
        '''Simulate an image with noise from a `imaging_pipeline`.'''

        # Combine two previously split PyTrees
        distribution = eqx.combine(dist_vmap, dist_novmap)

        return distribution.sample(key)

    write_simulated_image_stack_from_starfile(
        metadata,
        compute_image_stack,
        imaging_pipeline,
        seed=0,
        overwrite=True,
    )
    ```

    **Arguments:**

    - `metadata`:
        The `RelionParticleMetadata` dataset.
    - `compute_image_stack`:
        A callable that computes the image stack from the parameters contained
        in the STAR file.
    - `pytree` :
        The pytree that is given to `compute_image_stack`
        to compute the image stack (before filtering for vmapping).
    - `args`:
        The arguments to pass to the `compute_image_stack` function.
    - `is_jittable`:
        Whether the `compute_image_stack` function is jittable with `equinox.filter_jit`.
    - `batch_size_per_mrc`:
        The maximum number of images that will be computed in each vmap operation.
        If `None`, all images for a single mrc file will be computed in a single vmap operation.
        Only relevant if `is_jittable` is True.
    - `seed`:
        The seed for the random number generator.
    - `overwrite`:
        Whether to overwrite the MRC files if they already exist.
    - `compression`:
        The compression to use when writing the MRC files.
    """  # noqa
    # Create the directory for the MRC files if it doesn't exist
    if not os.path.exists(metadata.path_to_relion_project):
        os.makedirs(metadata.path_to_relion_project)

    else:
        mrc_fnames = (
            metadata.data_blocks["particles"]["rlnImageName"]
            .str.split("@", expand=True)[1]
            .unique()
        )

        if not overwrite:
            for mrc_fname in mrc_fnames:
                filename = os.path.join(metadata.path_to_relion_project, mrc_fname)
                if os.path.exists(filename):
                    raise FileExistsError(
                        f"Overwrite was set to False,\
                            but MRC file {filename} in starfile already exists."
                    )
        else:
            # remove existing MRC files if they match with the ones in the starfile
            for mrc_fname in mrc_fnames:
                filename = os.path.join(metadata.path_to_relion_project, mrc_fname)
                if os.path.exists(filename):
                    os.remove(filename)

    if is_jittable:
        _write_simulated_image_stack_from_starfile_vmap(
            metadata=metadata,
            compute_image=compute_image,
            args=args,
            batch_size_per_mrc=batch_size_per_mrc,
            seed=seed,
            overwrite=overwrite,
            compression=compression,
        )

    else:
        _write_simulated_image_stack_from_starfile_serial(
            metadata=metadata,
            compute_image=compute_image,
            args=args,
            seed=seed,
            overwrite=overwrite,
            compression=compression,
        )

    return


def _compute_images_batch(
    indices, metadata, compute_image_stack, args, filter_spec_for_vmap, key=None
):
    relion_particle_metadata = metadata[indices]
    # ... split the particle stack based on parameters to vmap over
    vmap, novmap = eqx.partition(relion_particle_metadata, filter_spec_for_vmap)
    # ... simulate images in the image stack
    if key is None:
        image_stack = compute_image_stack(vmap, novmap, args)

    else:
        # ... generate keys for each image in the mrcfile,
        # and a subkey for the next iteration

        key, *subkeys = jax.random.split(key, len(indices) + 1)
        subkeys = jax.numpy.array(subkeys)

        image_stack = compute_image_stack(
            subkeys,
            vmap,
            novmap,
            args,  # type: ignore
        )
    return image_stack, key


def _write_simulated_image_stack_from_starfile_vmap(
    metadata: RelionParticleMetadata,
    compute_image: (
        Callable[[RelionParticleParameters, Any], Float[Array, "y_dim x_dim"]]
        | Callable[
            [PRNGKeyArray, RelionParticleParameters, Any],
            Float[Array, "y_dim x_dim"],
        ]
    ),
    args: Any,
    batch_size_per_mrc: Optional[int] = None,
    seed: Optional[int] = None,  # seed for the noise
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    # Create RNG key, along with a subkey for subsequent use
    if seed is not None:
        key = jax.random.PRNGKey(seed=seed)
    else:
        key = cast(PRNGKeyArray, None)
    # Create vmapped `compute_image` kernel
    test_particle_metadata = metadata[0]
    filter_spec_for_vmap = _get_particle_metadata_filter_spec(test_particle_metadata)
    compute_image_stack = (
        eqx.filter_vmap(
            lambda vmap, novmap, args: compute_image(eqx.combine(vmap, novmap), args),  # type: ignore
            in_axes=(0, None, None),
        )
        if seed is None
        else eqx.filter_vmap(
            lambda key, vmap, novmap, args: compute_image(
                key, eqx.combine(vmap, novmap), args
            ),  # type: ignore
            in_axes=(0, 0, None, None),
        )
    )
    compute_image_stack = eqx.filter_jit(compute_image_stack)

    # check if function runs
    vmap, novmap = eqx.partition(metadata[0:2], filter_spec_for_vmap)
    if seed is None:
        try:
            compute_image_stack(vmap, novmap, args)
        except Exception as e:
            raise RuntimeError(
                "The `compute_image` function failed to run.\
                    Please check the function signature and arguments.\
                        Confirm that your function is jittable if necessary."
            ) from e
    else:
        key, *subkeys = jax.random.split(key, 3)
        try:
            compute_image_stack(jnp.array(subkeys), vmap, novmap, args)
        except Exception as e:
            raise RuntimeError(
                "The `compute_image` function failed to run.\
                    Please check the function signature and arguments.\
                        Confirm that your function is jittable if necessary."
            ) from e

    # Now, let's preparing the simulation loop. First check how many unique MRC
    # files we have in the starfile
    particles_fnames = metadata.data_blocks["particles"]["rlnImageName"].str.split(
        "@", expand=True
    )
    mrc_fnames = particles_fnames[1].unique()

    box_size = int(metadata.data_blocks["optics"]["rlnImageSize"][0])
    pixel_size = float(metadata.data_blocks["optics"]["rlnImagePixelSize"][0])

    # ... now, generate images for each mrcfile
    for mrc_fname in mrc_fnames:
        # ... check which indices in the starfile correspond to this mrc file
        # and load the particle stack parameters
        indices = particles_fnames[particles_fnames[1] == mrc_fname].index.to_numpy()

        if batch_size_per_mrc is not None:
            n_batches = max(len(indices) // batch_size_per_mrc, 1)
            residual = len(indices) % batch_size_per_mrc if n_batches > 1 else 0

        else:
            n_batches = 1
            residual = 0

        if n_batches > 1:
            image_stack = np.empty((len(indices), box_size, box_size), dtype=np.float32)

            for i in range(n_batches):
                idx_0 = i * batch_size_per_mrc
                idx_1 = (i + 1) * batch_size_per_mrc
                image_stack[idx_0:idx_1], key = _compute_images_batch(
                    indices[idx_0:idx_1],
                    metadata,
                    compute_image_stack,
                    args,
                    filter_spec_for_vmap,
                    key=key,
                )

            if residual > 0:
                image_stack[-residual:], key = _compute_images_batch(
                    indices[-residual:],
                    metadata,
                    compute_image_stack,
                    args,
                    filter_spec_for_vmap,
                    key=key,
                )

            image_stack = jnp.array(image_stack)

        else:
            image_stack, key = _compute_images_batch(
                indices,
                metadata,
                compute_image_stack,
                args,
                filter_spec_for_vmap,
                key=key,
            )

        # ... write the image stack to an MRC file
        filename = os.path.join(metadata.path_to_relion_project, mrc_fname)
        write_image_stack_to_mrc(
            image_stack,
            pixel_size=pixel_size,
            filename=filename,
            overwrite=overwrite,
            compression=compression,
        )

    return


def _write_simulated_image_stack_from_starfile_serial(
    metadata: RelionParticleMetadata,
    compute_image: (
        Callable[[RelionParticleParameters, Any], Float[Array, "y_dim x_dim"]]
        | Callable[
            [PRNGKeyArray, RelionParticleParameters, Any],
            Float[Array, "y_dim x_dim"],
        ]
    ),
    args: Any,
    seed: Optional[int] = None,  # seed for the noise
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    # Create RNG key, along with a subkey for subsequent use
    if seed is not None:
        key = jax.random.PRNGKey(seed=seed)
    else:
        key = cast(PRNGKeyArray, None)

    # Now, let's preparing the simulation loop. First check how many unique MRC
    # files we have in the starfile
    particles_fnames = metadata.data_blocks["particles"]["rlnImageName"].str.split(
        "@", expand=True
    )
    mrc_fnames = particles_fnames[1].unique()

    box_size = int(metadata.data_blocks["optics"]["rlnImageSize"][0])
    pixel_size = float(metadata.data_blocks["optics"]["rlnImagePixelSize"][0])

    # ... now, generate images for each mrcfile
    for mrc_fname in mrc_fnames:
        # ... check which indices in the starfile correspond to this mrc file
        # and load the particle stack parameters
        indices = particles_fnames[particles_fnames[1] == mrc_fname].index.to_numpy()
        image_stack = np.empty((len(indices), box_size, box_size), dtype=np.float32)

        for i in range(len(indices)):
            if seed is not None:
                key, subkey = jax.random.split(key)
                image_stack[i] = compute_image(subkey, metadata[indices[i]], args)

            else:
                image_stack[i] = compute_image(metadata[indices[i]], args)

            # ... write the image stack to an MRC file
            filename = os.path.join(metadata.path_to_relion_project, mrc_fname)
            write_image_stack_to_mrc(
                image_stack,
                pixel_size=pixel_size,
                filename=filename,
                overwrite=overwrite,
                compression=compression,
            )

    return


_get_particle_metadata_filter_spec = lambda particle_metadata: get_filter_spec(
    particle_metadata, _pointer_to_vmapped_parameters
)


def _pointer_to_vmapped_parameters(particle_metadata):
    if isinstance(particle_metadata.transfer_theory.envelope, FourierGaussian):
        output = (
            particle_metadata.transfer_theory.ctf.defocus_in_angstroms,
            particle_metadata.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_metadata.transfer_theory.ctf.astigmatism_angle,
            particle_metadata.transfer_theory.ctf.phase_shift,
            particle_metadata.transfer_theory.envelope.b_factor,
            particle_metadata.transfer_theory.envelope.amplitude,
            particle_metadata.pose,
        )
    else:
        output = (
            particle_metadata.transfer_theory.ctf.defocus_in_angstroms,
            particle_metadata.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_metadata.transfer_theory.ctf.astigmatism_angle,
            particle_metadata.transfer_theory.ctf.phase_shift,
            particle_metadata.pose,
        )
    return output
