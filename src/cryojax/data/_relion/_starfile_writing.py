import os
import pathlib
from typing import Callable, Optional, TypeVar

import equinox as eqx
import jax
import numpy as np
import pandas as pd
import starfile
from jaxtyping import Array, Float

from ...image.operators import Constant, FourierGaussian
from ...io import write_image_stack_to_mrc
from ...simulator import AberratedAstigmaticCTF
from ...utils import batched_map
from ._starfile_dataset import RelionParticleParameterDataset
from ._starfile_pytrees import RelionParticleParameters


PerParticlePyTree = TypeVar("PerParticlePyTree")
ConstantPyTree = TypeVar("ConstantPyTree")


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
    """Generate a STAR file from a `RelionParticleParameters` object.

    This function does not generate particles, it merely populates a starfile
    and writes it to disc at the location specified by filename.

    **Arguments:**

    - `particle_parameters`:
        The `RelionParticleParameters` object.
    - `filename`:
        The filename of the STAR file to write.
    - `mrc_batch_size`:
        The number of images to write to each MRC file. If `None`, the number of
        images in the `RelionParticleParameters` is used.
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

    if particle_parameters.pose.offset_x_in_angstroms.shape == ():
        n_images = 1
    else:
        n_images = particle_parameters.pose.offset_x_in_angstroms.shape[0]

    if mrc_batch_size is None:
        mrc_batch_size = n_images

    assert (
        n_images >= mrc_batch_size
    ), "n_images must be greater than or equal to mrc_batch_size"

    starfile_dict = dict()
    if not isinstance(particle_parameters.transfer_theory.ctf, AberratedAstigmaticCTF):
        raise NotImplementedError(
            "The `RelionParticleParameters.transfer_theory.ctf` must be an "
            "`AberratedAstigmaticCTF`. Found that it was a "
            f"{type(particle_parameters.transfer_theory.ctf).__name__}."
        )
    # Generate optics group
    optics_df = pd.DataFrame()
    optics_df["rlnOpticsGroup"] = [1]
    optics_df["rlnVoltage"] = particle_parameters.instrument_config.voltage_in_kilovolts[
        0
    ]
    optics_df["rlnSphericalAberration"] = (
        particle_parameters.transfer_theory.ctf.spherical_aberration_in_mm[0]
    )
    optics_df["rlnImagePixelSize"] = particle_parameters.instrument_config.pixel_size[0]
    optics_df["rlnImageSize"] = particle_parameters.instrument_config.shape[0]
    optics_df["rlnAmplitudeContrast"] = (
        particle_parameters.transfer_theory.amplitude_contrast_ratio[0]
    )
    starfile_dict["optics"] = optics_df

    # Generate particles group
    particles_df = pd.DataFrame()

    # Fixed value parameters
    particles_df["rlnCtfMaxResolution"] = np.zeros(n_images)
    particles_df["rlnCtfFigureOfMerit"] = np.zeros(n_images)
    particles_df["rlnClassNumber"] = np.ones(n_images)
    particles_df["rlnOpticsGroup"] = np.ones(n_images)
    # Pose (flipping the sign of the translations); RELION's convention
    # thinks about "undoing" a translation, opposed to simulating an image at a coordinate
    particles_df["rlnOriginXAngst"] = -particle_parameters.pose.offset_x_in_angstroms
    particles_df["rlnOriginYAngst"] = -particle_parameters.pose.offset_y_in_angstroms
    particles_df["rlnAngleRot"] = -particle_parameters.pose.phi_angle
    particles_df["rlnAngleTilt"] = -particle_parameters.pose.theta_angle
    particles_df["rlnAnglePsi"] = -particle_parameters.pose.psi_angle
    # CTF
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
    elif particle_parameters.transfer_theory.envelope is None:
        particles_df["rlnCtfBfactor"] = 0.0
        particles_df["rlnCtfScalefactor"] = 0.0
    else:
        raise NotImplementedError(
            "The envelope function in `RelionParticleParameters` must either be "
            "`cryojax.image.operators.FourierGaussian` or "
            "`cryojax.image.operators.Constant`. Got "
            f"{type(particle_parameters.transfer_theory.envelope).__name__}."
        )

    particles_df["rlnPhaseShift"] = particle_parameters.transfer_theory.phase_shift

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
    param_dataset: RelionParticleParameterDataset,
    compute_image_fn: Callable[
        [RelionParticleParameters, ConstantPyTree, PerParticlePyTree],
        Float[Array, "y_dim x_dim"],
    ],
    constant_args: ConstantPyTree = None,
    per_particle_args: PerParticlePyTree = None,
    is_jittable: bool = False,
    batch_size_per_mrc: Optional[int] = None,
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    """Writes a stack of images from the parameters contained in a
    STAR file.

    !!! note
        This function works generally for a function that computes
        images of the form `compute_image_fn(parameter_dataset, constant_args, per_particle_args)`
        where `parameter_dataset`is a cryojax RelionParticleParameterDataset,
        `constant_args` is a parameter that does not change between images
        , and `per_particle_args` is a pytree whose leaves have a batch dimension
        equal to the number of particles to be simulated.

    ```python
    # Example 1: Using the function with a `compute_image_fn`
    # function that does not use `per_particle_args`

    # load a `RelionParticleParameterDataset` object
    param_dataset = RelionParticleParameterDataset(...)

    # to build an`image_model` object
    # (see our Tutorials for details on how to do this)

    # Write your `compute_image_fn` function.
    def compute_image_fn(
        particle_parameters: RelionParticleParameters,
        constant_args
    ):

        # constant_args do not change between images
        potential_integrator, ... = constant_args # Just an example

        # use the pose, ctf, etc from the particle_parameters
        img_pipeline = ContrastImageModel(...)

        return img_pipeline.render()

    write_simulated_image_stack_from_starfile(
        param_dataset, # contains particle_parameters objects
        compute_image_fn,
        constant_args=(potential_integrator, ...)
        per_particle_args=None, # default
        is_jittable=True,
        overwrite=True,
    )

    ```

    ```python
    # Example 2: Using the function with a `compute_image_fn`
    # function that uses per_particle arguments

    # now we will build a cryojax `distribution` and use it
    # to generate noisy images.
    # (see our Tutorials for details on how to do this)

    # Let's define a `RelionParticleParameterDataset` object

    param_dataset = RelionParticleParameterDataset(...)

    # and the rng keys to generate the noise for the images
    seed = 0
    key = jax.random.key(seed)
    key, *keys_noise = jax.random.split(key, n_images+1)
    keys_noise = jnp.array(keys_noise)

    # In addition, we will add a scaling parameter to our images

    key, subkey = jax.random.split(key)
    scaling_params = jax.random.uniform(subkey, shape=(n_images,))

    # Write your `compute_image_fn` function
    def compute_noisy_image_stack(
        particle_parameters: RelionParticleParameters,
        constant_args,
        per_particle_args
    ):
        ... = constant_args
        key, scale = per_particle_args

        # Combine two previously split PyTrees
        image_model = cxs.ContrastImageModel(...)
        distribution = cxs.IndependentGaussianPixels(image_model, ...)

        return scale * distribution.sample(key)

    write_simulated_image_stack_from_starfile(
        param_dataset, # contains particle_parameters objects
        compute_image_fn,
        constant_args=(...)
        per_particle_args=(keys_noise, scaling_params)
        is_jittable=True,
        overwrite=True,
    )
    ```

    **Arguments:**

    - `param_dataset`:
        The `RelionParticleParameterDataset` dataset.
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
    - `is_jittable`:
        Whether the `compute_image_fn` function is jittable with `equinox.filter_jit`.
    - `batch_size_per_mrc`:
        The maximum number of images that will be computed in each vmap operation.
        If `None`, all images for a single mrc file will be computed in a single vmap operation.
        Only relevant if `is_jittable` is True.
    - `overwrite`:
        Whether to overwrite the MRC files if they already exist.
    - `compression`:
        The compression to use when writing the MRC files.
    """  # noqa

    # Create the directory for the MRC files if it doesn't exist
    if not os.path.exists(param_dataset.path_to_relion_project):
        os.makedirs(param_dataset.path_to_relion_project)

    else:
        mrc_fnames = (
            param_dataset.starfile_data["particles"]["rlnImageName"]
            .str.split("@", expand=True)[1]
            .unique()
        )

        if not overwrite:
            for mrc_fname in mrc_fnames:
                filename = os.path.join(param_dataset.path_to_relion_project, mrc_fname)
                if os.path.exists(filename):
                    raise FileExistsError(
                        f"Overwrite was set to False,\
                            but MRC file {filename} in starfile already exists."
                    )
        else:
            # remove existing MRC files if they match with the ones in the starfile
            for mrc_fname in mrc_fnames:
                filename = os.path.join(param_dataset.path_to_relion_project, mrc_fname)
                if os.path.exists(filename):
                    os.remove(filename)

    if per_particle_args is not None:
        shape_check = jax.tree.map(
            lambda x: eqx.error_if(
                x=x,
                pred=x.shape[0] != len(param_dataset),
                msg="All leaves of per_particle_args should have"
                + "a batch dimension equal to the number of images",
            ),
            per_particle_args,
        )

        del shape_check

    if is_jittable:
        _write_simulated_image_stack_from_starfile_vmap(
            param_dataset=param_dataset,
            compute_image_fn=compute_image_fn,
            constant_args=constant_args,
            per_particle_args=per_particle_args,
            batch_size_per_mrc=batch_size_per_mrc,
            overwrite=overwrite,
            compression=compression,
        )

    else:
        _write_simulated_image_stack_from_starfile_serial(
            param_dataset=param_dataset,
            compute_image_fn=compute_image_fn,
            constant_args=constant_args,
            per_particle_args=per_particle_args,
            overwrite=overwrite,
            compression=compression,
        )

    return


def _write_simulated_image_stack_from_starfile_vmap(
    param_dataset: RelionParticleParameterDataset,
    compute_image_fn: Callable[
        [RelionParticleParameters, ConstantPyTree, PerParticlePyTree],
        Float[Array, "y_dim x_dim"],
    ],
    constant_args: ConstantPyTree,
    per_particle_args: PerParticlePyTree,
    batch_size_per_mrc: Optional[int] = None,
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    # Create vmapped `compute_image` kernel
    compute_image_stack = eqx.filter_vmap(
        compute_image_fn,
        in_axes=(eqx.if_array(0), None, eqx.if_array(0)),
    )
    compute_image_stack = eqx.filter_jit(compute_image_stack)

    # check if function runs
    test_particle_parameters = param_dataset[0:1]
    try:
        compute_image_stack(
            test_particle_parameters,
            constant_args,
            jax.tree.map(lambda x: x[0:1], per_particle_args),
        )
    except Exception as e:
        raise RuntimeError(
            "The `compute_image` function failed to run.\
                Please check the function signature and arguments.\
                    Confirm that your function is jittable if necessary."
        ) from e

    # Now, let's preparing the simulation loop. First check how many unique MRC
    # files we have in the starfile
    particles_fnames = param_dataset.starfile_data["particles"]["rlnImageName"].str.split(
        "@", expand=True
    )
    mrc_fnames = particles_fnames[1].unique()
    pixel_size = float(param_dataset.starfile_data["optics"]["rlnImagePixelSize"][0])

    # ... now, generate images for each mrcfile
    for mrc_fname in mrc_fnames:
        # ... check which indices in the starfile correspond to this mrc file
        # and load the particle stack parameters
        indices = particles_fnames[particles_fnames[1] == mrc_fname].index.to_numpy()

        if batch_size_per_mrc is None:
            batch_size_for_map = len(indices)
        else:
            batch_size_for_map = min(batch_size_per_mrc, len(indices))

        vmap, novmap = eqx.partition(param_dataset[indices], eqx.is_array)
        # image_stack = batched_map(
        #     lambda x: _compute_image_stack_map_wrapper(
        #         compute_image_stack, x[0], novmap, constant_args, x[1]
        #     ),
        #     xs=(
        #         vmap,  # type: ignore
        #         jax.tree.map(lambda x: x[indices], per_particle_args),  # type: ignore
        #     ),
        #     batch_size=batch_size_for_map,
        # )

        image_stack = batched_map(
            lambda x: compute_image_stack(eqx.combine(x[0], novmap), constant_args, x[1]),
            xs=(
                vmap,
                jax.tree.map(lambda x: x[indices], per_particle_args),
            ),
            batch_size=batch_size_for_map,
        )

        # ... write the image stack to an MRC file
        filename = os.path.join(param_dataset.path_to_relion_project, mrc_fname)
        write_image_stack_to_mrc(
            image_stack,
            pixel_size=pixel_size,
            filename=filename,
            overwrite=overwrite,
            compression=compression,
        )

    return


def _write_simulated_image_stack_from_starfile_serial(
    param_dataset: RelionParticleParameterDataset,
    compute_image_fn: Callable[
        [RelionParticleParameters, ConstantPyTree, PerParticlePyTree],
        Float[Array, "y_dim x_dim"],
    ],
    constant_args: ConstantPyTree,
    per_particle_args: PerParticlePyTree,
    overwrite: bool = False,
    compression: Optional[str] = None,
):
    # Now, let's preparing the simulation loop. First check how many unique MRC
    # files we have in the starfile
    particles_fnames = param_dataset.starfile_data["particles"]["rlnImageName"].str.split(
        "@", expand=True
    )
    mrc_fnames = particles_fnames[1].unique()

    box_size = int(param_dataset.starfile_data["optics"]["rlnImageSize"][0])
    pixel_size = float(param_dataset.starfile_data["optics"]["rlnImagePixelSize"][0])

    # ... now, generate images for each mrcfile
    for mrc_fname in mrc_fnames:
        # ... check which indices in the starfile correspond to this mrc file
        # and load the particle stack parameters
        indices = particles_fnames[particles_fnames[1] == mrc_fname].index.to_numpy()
        image_stack = np.empty((len(indices), box_size, box_size), dtype=np.float32)

        for i in range(len(indices)):
            image_stack[i] = compute_image_fn(
                param_dataset[indices[i]],
                constant_args,
                jax.tree.map(lambda x: x[indices[i]], per_particle_args),  # type: ignore
            )

        # ... write the image stack to an MRC file
        filename = os.path.join(param_dataset.path_to_relion_project, mrc_fname)
        write_image_stack_to_mrc(
            image_stack,
            pixel_size=pixel_size,
            filename=filename,
            overwrite=overwrite,
            compression=compression,
        )

    return
