import os
import pathlib
from typing import Any, Callable, cast, Optional

import equinox as eqx
import jax
import numpy as np
import pandas as pd
import starfile
from jaxtyping import Array, Float, PRNGKeyArray

from ..._filter_specs import get_filter_spec
from ...image.operators import Constant, FourierGaussian
from ...io import write_image_stack_to_mrc
from ._starfile_reading import RelionDataset, RelionParticleStack


def _get_filename(step, n_char=6):
    if step == 0:
        fname = "0" * n_char
    else:
        n_dec = int(np.log10(step))
        fname = "0" * (n_char - n_dec) + str(step)
    return fname


def write_starfile_with_particle_parameters(
    relion_particle_stack: RelionParticleStack,
    filename: str | pathlib.Path,
    mrc_batch_size: Optional[int] = None,
) -> None:
    """Generate a STAR file from a RelionParticleStack object.

    This function does not generate particles, it merely populates the starfile.

    The starfile is written to disc at the location specified by filename.

    **Arguments:**

    - `relion_particle_stack`:
        The `RelionParticleStack` object.
    - `filename`:
        The filename of the STAR file to write.
    - `mrc_batch_size`:
        The number of images to write to each MRC file. If `None`, the number of
        images in the `RelionParticleStack` is used.
    """

    n_images = relion_particle_stack.pose.offset_x_in_angstroms.shape[0]

    if mrc_batch_size is None:
        mrc_batch_size = n_images

    assert (
        n_images >= mrc_batch_size
    ), "n_images must be greater than or equal to mrc_batch_size"

    starfile_dict = dict()
    # Generate optics group
    optics_df = pd.DataFrame()
    optics_df["rlnOpticsGroup"] = [1]
    optics_df["rlnVoltage"] = relion_particle_stack.instrument_config.voltage_in_kilovolts
    optics_df["rlnSphericalAberration"] = (
        relion_particle_stack.transfer_theory.ctf.spherical_aberration_in_mm
    )
    optics_df["rlnImagePixelSize"] = relion_particle_stack.instrument_config.pixel_size
    optics_df["rlnImageSize"] = relion_particle_stack.instrument_config.shape[0]
    optics_df["rlnAmplitudeContrast"] = (
        relion_particle_stack.transfer_theory.ctf.amplitude_contrast_ratio
    )
    starfile_dict["optics"] = optics_df

    # Generate particles group
    particles_df = pd.DataFrame()

    # fixed values
    particles_df["rlnCtfMaxResolution"] = np.zeros(n_images)
    particles_df["rlnCtfFigureOfMerit"] = np.zeros(n_images)
    particles_df["rlnClassNumber"] = np.ones(n_images)
    particles_df["rlnOpticsGroup"] = np.ones(n_images)

    particles_df["rlnOriginXAngst"] = relion_particle_stack.pose.offset_x_in_angstroms
    particles_df["rlnOriginYAngst"] = relion_particle_stack.pose.offset_y_in_angstroms
    particles_df["rlnAngleRot"] = relion_particle_stack.pose.view_phi
    particles_df["rlnAngleTilt"] = relion_particle_stack.pose.view_theta
    particles_df["rlnAnglePsi"] = relion_particle_stack.pose.view_psi
    particles_df["rlnDefocusU"] = (
        relion_particle_stack.transfer_theory.ctf.defocus_in_angstroms
    )
    particles_df["rlnDefocusV"] = (
        relion_particle_stack.transfer_theory.ctf.astigmatism_in_angstroms
    )
    particles_df["rlnDefocusAngle"] = (
        relion_particle_stack.transfer_theory.ctf.astigmatism_angle
    )

    if isinstance(relion_particle_stack.transfer_theory.envelope, FourierGaussian):
        particles_df["rlnCtfBfactor"] = (
            relion_particle_stack.transfer_theory.envelope.b_factor
        )
        particles_df["rlnCtfScalefactor"] = (
            relion_particle_stack.transfer_theory.envelope.amplitude
        )

    elif isinstance(relion_particle_stack.transfer_theory.envelope, Constant):
        particles_df["rlnCtfBfactor"] = 0.0
        particles_df["rlnCtfScalefactor"] = (
            relion_particle_stack.transfer_theory.envelope.value
        )

    else:
        raise NotImplementedError(
            "Only FourierGaussian and Constant envelopes are supported"
        )

    particles_df["rlnPhaseShift"] = relion_particle_stack.transfer_theory.ctf.phase_shift

    n_batches = n_images // mrc_batch_size
    n_remainder = n_images % mrc_batch_size

    relative_mrcs_path_prefix = str(filename).split(".")[0]
    image_names = []

    for step in range(n_batches):
        mrc_filename = _get_filename(step, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + mrc_filename + ".mrcs"
        image_names += [
            _get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(mrc_batch_size)
        ]

    if n_remainder > 0:
        mrc_filename = _get_filename(n_batches, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + mrc_filename + ".mrcs"
        image_names += [
            _get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(n_remainder)
        ]

    particles_df["rlnImageName"] = image_names
    starfile_dict["particles"] = particles_df
    starfile.write(starfile_dict, pathlib.Path(filename))

    return


def write_simulated_image_stack_from_starfile(
    dataset: RelionDataset,
    compute_image: (
        Callable[[RelionParticleStack, Any], Float[Array, "y_dim x_dim"]]
        | Callable[
            [PRNGKeyArray, RelionParticleStack, Any],
            Float[Array, "y_dim x_dim"],
        ]
    ),
    args: Any,
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

    # and a dataset
    dataset = RelionDataset(...)

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
        dataset,
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

    distribution =cryojax.inference.IndependentGaussianFourierModes

    # and a dataset
    dataset = RelionDataset(...)

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
        dataset,
        compute_image_stack,
        imaging_pipeline,
        seed=0,
        overwrite=True,
    )
    ```

    **Arguments:**

    - `dataset`:
        The `RelionDataset` STAR file reader.
    - `compute_image_stack`:
        A callable that computes the image stack from the parameters contained
        in the STAR file.
    - `pytree` :
        The pytree that is given to `compute_image_stack`
        to compute the image stack (before filtering for vmapping).
    - `seed`:
        The seed for the random number generator.
    - `overwrite`:
        Whether to overwrite the MRC files if they already exist.
    - `compression`:
        The compression to use when writing the MRC files.
    """
    # Create the directory for the MRC files if it doesn't exist
    if not os.path.exists(dataset.path_to_relion_project):
        os.makedirs(dataset.path_to_relion_project)
    # Create RNG key, along with a subkey for subsequent use
    if seed is not None:
        key = jax.random.PRNGKey(seed=seed)
    else:
        key = cast(PRNGKeyArray, None)
    # Create vmapped `compute_image` kernel
    test_particle_stack = dataset[0]
    filter_spec_for_vmap = _get_particle_stack_filter_spec(test_particle_stack)
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

    # Try to jit the function
    try:
        compute_image_stack = eqx.filter_jit(compute_image_stack)
    except Any:
        pass

    # Now, let's preparing the simulation loop. First check how many unique MRC
    # files we have in the starfile
    particles_fnames = dataset.data_blocks["particles"]["rlnImageName"].str.split(
        "@", expand=True
    )
    mrc_fnames = particles_fnames[1].unique()
    # ... now, generate images for each mrcfile
    for mrc_fname in mrc_fnames:
        # ... check which indices in the starfile correspond to this mrc file
        # and load the particle stack parameters
        indices = particles_fnames[particles_fnames[1] == mrc_fname].index.to_numpy()
        relion_particle_stack = dataset[indices]
        # ... split the particle stack based on parameters to vmap over
        vmap, novmap = eqx.partition(relion_particle_stack, filter_spec_for_vmap)
        # ... simulate images in the image stack
        if seed is None:
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

        # ... write the image stack to an MRC file
        filename = os.path.join(dataset.path_to_relion_project, mrc_fname)
        write_image_stack_to_mrc(
            image_stack,
            pixel_size=relion_particle_stack.instrument_config.pixel_size,
            filename=filename,
            overwrite=overwrite,
            compression=compression,
        )

    return


_get_particle_stack_filter_spec = lambda particle_stack: get_filter_spec(
    particle_stack, _pointer_to_vmapped_parameters
)


def _pointer_to_vmapped_parameters(particle_stack):
    if isinstance(particle_stack.transfer_theory.envelope, FourierGaussian):
        output = (
            particle_stack.transfer_theory.ctf.defocus_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_angle,
            particle_stack.transfer_theory.ctf.phase_shift,
            particle_stack.transfer_theory.envelope.b_factor,
            particle_stack.transfer_theory.envelope.amplitude,
            particle_stack.pose.offset_x_in_angstroms,
            particle_stack.pose.offset_y_in_angstroms,
            particle_stack.pose.view_phi,
            particle_stack.pose.view_theta,
            particle_stack.pose.view_psi,
        )
    else:
        output = (
            particle_stack.transfer_theory.ctf.defocus_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_angle,
            particle_stack.transfer_theory.ctf.phase_shift,
            particle_stack.pose.offset_x_in_angstroms,
            particle_stack.pose.offset_y_in_angstroms,
            particle_stack.pose.view_phi,
            particle_stack.pose.view_theta,
            particle_stack.pose.view_psi,
        )
    return output
