
import os
import mrcfile
import starfile
import numpy as np
import pandas as pd
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import config
from jaxlie import SO3
from functools import partial
from jax.scipy.spatial.transform import Rotation

import cryojax.simulator as cs
from cryojax.io import read_array_with_spacing_from_mrc

config.update("jax_enable_x64", False)

def gen_specimen_instrument_random_poses(n_poses, density, image_size):
    # Sample a group of uniform rotations and translations
    keys = jax.random.split(jax.random.PRNGKey(0), n_poses)
    # ... rotations
    rotations = jax.vmap(lambda key: SO3.sample_uniform(key))(keys)
    # ... translations
    translations = jax.vmap(
        lambda key: jax.random.uniform(
            key, (3,), minval=-image_size / 4, maxval=image_size / 4
        )
    )(keys)
    translations /= jnp.asarray([1.0, 1.0, image_size / 4])

    # Now, instantiate the poses
    poses = jax.vmap(lambda R, t: cs.QuaternionPose.from_rotation_and_translation(R, t))(
        rotations, translations
    )

    # ... build the specimen
    specimen = cs.Specimen(density, poses)

    # ... and finally the instrument
    optics = cs.CTFOptics()
    instrument = cs.Instrument(optics=optics)

    return (specimen, instrument)

@partial(jax.vmap, in_axes=[0, None])
def compute_image(vmap, novmap):
    """Compute image stack."""
    pipeline = eqx.combine(vmap, novmap)
    return pipeline.render()

@jax.jit
def compute_image_stack(specimen, scattering, instrument):
    """Build the model and compute the superposition."""
    pipeline = cs.ImagePipeline(specimen, scattering, instrument)
    is_vmap = lambda x: isinstance(x, cs.AbstractPose)
    to_vmap = jax.tree_util.tree_map(is_vmap, pipeline, is_leaf=is_vmap)
    vmap, novmap = eqx.partition(pipeline, to_vmap)
    return compute_image(vmap, novmap)

def get_filename(step, n_char=6):
    if step == 0:
        return "0" * n_char
    else:
        n_dec = int(np.log10(step))
        return "0" * (n_char - n_dec) + str(step)

def create_df_for_starfile_(starfile_fname, n_images, batch_size):

    starf_new = dict()

    # Generate optics group
    optics_df = pd.DataFrame()
    optics_df["rlnOpticsGroup"] = [1]
    optics_df["rlnVoltage"] = [0]
    optics_df["rlnSphericalAberration"] = [0]
    optics_df["rlnImagePixelSize"] = [0]
    optics_df["rlnImageSize"] = [0]
    optics_df["rlnAmplitudeContrast"] = [0]
    starf_new["optics"] = optics_df

    # Generate particles group
    particles_df = pd.DataFrame()
    particles_df["rlnOriginXAngst"] = np.zeros(n_images)
    particles_df["rlnOriginYAngst"] = np.zeros(n_images)
    particles_df["rlnAngleRot"] = np.zeros(n_images)
    particles_df["rlnAngleTilt"] = np.zeros(n_images)
    particles_df["rlnAnglePsi"] = np.zeros(n_images)
    particles_df["rlnDefocusU"] = np.zeros(n_images)
    particles_df["rlnDefocusV"] = np.zeros(n_images)
    particles_df["rlnDefocusAngle"] = np.zeros(n_images)
    particles_df["rlnCtfBfactor"] = np.zeros(n_images)
    particles_df["rlnCtfScalefactor"] = np.zeros(n_images)
    particles_df["rlnPhaseShift"] = np.zeros(n_images)

    # fixed values
    particles_df["rlnCtfMaxResolution"] = np.zeros(n_images)
    particles_df["rlnCtfFigureOfMerit"] = np.zeros(n_images)
    particles_df["rlnRandomSubset"] = np.random.randint(1, 2, size=n_images)
    particles_df["rlnClassNumber"] = np.ones(n_images)
    particles_df["rlnOpticsGroup"] = np.ones(n_images)

    n_batches = n_images // batch_size
    n_remainder = n_images % batch_size

    relative_mrcs_path_prefix = starfile_fname.split(".")[0]
    image_names = []

    for step in range(n_batches):
        filename = get_filename(step, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + filename + ".mrcs"
        image_names += [
            get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(batch_size)
        ]

    if n_remainder > 0:
        filename = get_filename(n_batches, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + filename + ".mrcs"
        image_names += [
            get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(n_remainder)
        ]

    particles_df["rlnImageName"] = image_names

    starf_new["particles"] = particles_df

    return starf_new

def update_starfile_particles(starf_df, specimen, instrument, init_idx, end_idx):

    angles = Rotation.from_quat(specimen.pose.wxyz).as_euler("ZYZ", degrees=True)

    starf_df["particles"].loc[init_idx:end_idx-1, "rlnOriginXAngst"] = specimen.pose.offset_x
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnOriginYAngst"] = specimen.pose.offset_y
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnAngleRot"] = angles[:, 0]
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnAngleTilt"] = angles[:, 1]
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnAnglePsi"] = angles[:, 2]
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnDefocusU"] = instrument.optics.ctf.defocus_u
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnDefocusV"] = instrument.optics.ctf.defocus_v
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnDefocusAngle"] = instrument.optics.ctf.defocus_angle
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnCtfBfactor"] = 0.0
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnCtfScalefactor"] = 1.0
    starf_df["particles"].loc[init_idx:end_idx-1, "rlnPhaseShift"] = instrument.optics.ctf.phase_shift

    return starf_df

def update_starfile_optics(starf_df, instrument, pixel_size, image_dim):

    starf_df["optics"].loc["rlnVoltage"] = instrument.optics.ctf.voltage
    starf_df["optics"].loc["rlnSphericalAberration"] = instrument.optics.ctf.spherical_aberration
    starf_df["optics"].loc["rlnImagePixelSize"] = pixel_size
    starf_df["optics"].loc["rlnImageSize"] = image_dim
    starf_df["optics"].loc["rlnAmplitudeContrast"] = instrument.optics.ctf.amplitude_contrast

    return starf_df

def write_mrcfile(mrc_path, images, n_particles, image_dim):

    with mrcfile.new_mmap(
        mrc_path,
        shape=(n_particles, image_dim, image_dim),
        mrc_mode=2,
        overwrite=True,
    ) as mrc_file:
        for j in range(n_particles):
            mrc_file.data[j] = images[j]

    return



def write_starfile(root_path, vol_fname, starfile_fname, n_images, batch_size, image_dim, pixel_size):
    """
    Write a starfile with the particles and the corresponding mrc files.

    Parameters
    ----------
    root_path : str
        Path to the directory where the starfile and the mrc files will be written.
    vol_fname : str
        Name of the mrc file containing the electron density (without path).
    starfile_fname : str
        Name of the starfile (without path).
    n_images : int
        Number of images to generate.
    batch_size : int
        Number of images per mrc file.
    image_dim : int
        Image size.
    pixel_size : float
        Pixel size in Angstroms.

    Returns
    -------
    None

    Writes a starfile and the corresponding mrc files in the root_path directory.
    """
    # ... load the ElectronDensity and ScatteringModel
    density_grid, voxel_size = read_array_with_spacing_from_mrc(vol_fname)
    config = cs.ImageConfig(shape, pixel_size, pad_scale=1.2)
    density = cs.FourierVoxelGrid.from_density_grid(density_grid, voxel_size, pad_scale=1.3)
    scattering = cs.FourierSliceExtract(config)

    # Configure the image parameters
    shape = (image_dim, image_dim)
    image_size = image_dim * pixel_size

    starfile_path = os.path.join(root_path, starfile_fname)
    new_starfile = create_df_for_starfile_(starfile_path, n_images, batch_size)

    n_batches = n_images // batch_size
    n_remainder = n_images % batch_size

    for i in range(n_batches):
        
        idx_init = i * batch_size
        idx_end = (i + 1) * batch_size

        mrc_relative_path = new_starfile["particles"]["rlnImageName"][
            batch_size * i
        ].split("@")[1]
        mrc_path = os.path.join(root_path, mrc_relative_path)

        # Get the random generator
        scattering, specimen, instrument = gen_specimen_instrument_random_poses(batch_size, density, image_size)
        images = compute_image_stack(specimen, scattering, instrument)

        new_starfile = update_starfile_particles(new_starfile, specimen, instrument, idx_init, idx_end)
        write_mrcfile(mrc_path, images, batch_size, image_dim)

    if n_remainder > 0:
        idx_init = n_batches * batch_size
        idx_end = n_images

        mrc_relative_path = new_starfile["particles"]["rlnImageName"][
            batch_size * n_batches
        ].split("@")[1]
        mrc_path = os.path.join(root_path, mrc_relative_path)

        # Get the random generator
        scattering, specimen, instrument = gen_specimen_instrument_random_poses(n_remainder, density, image_size)
        images = compute_image_stack(specimen, scattering, instrument)

        new_starfile = update_starfile_particles(new_starfile, specimen, instrument, idx_init, idx_end)
        write_mrcfile(mrc_path, images, n_remainder, image_dim)

    new_starfile = update_starfile_optics(new_starfile, instrument, pixel_size, image_dim)
    starfile.write(new_starfile, starfile_path)

    return

 