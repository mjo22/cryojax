import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import starfile

from ..data._relion import RelionParticleStack
from ..image.operators import Constant, FourierGaussian


def get_filename(step, n_char=6):
    if step == 0:
        fname = "0" * n_char
    else:
        n_dec = int(np.log10(step))
        fname = "0" * (n_char - n_dec) + str(step)
    return fname


def generate_starfile(
    relion_particle_stack: RelionParticleStack,
    filename: str | pathlib.Path,
    mrc_batch_size: Optional[int] = None,
) -> None:
    """
    Generate a STAR file from a RelionParticleStack object.

    This function does not generate particles, it merely populates the starfile.

    The starfile is written to disc at the location specified by filename.

    Parameters
    ----------
    relion_particle_stack : RelionParticleStack
        A RelionParticleStack object.
    filename : str
        The filename of the STAR file to write.
    mrc_batch_size : int, optional
        The number of images to write to each MRC file. If None, defaults to n_images.

    Returns
    -------
    None
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

    relative_mrcs_path_prefix = filename.split(".")[0]
    image_names = []

    for step in range(n_batches):
        mrc_filename = get_filename(step, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + mrc_filename + ".mrcs"
        image_names += [
            get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(mrc_batch_size)
        ]

    if n_remainder > 0:
        mrc_filename = get_filename(n_batches, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + mrc_filename + ".mrcs"
        image_names += [
            get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(n_remainder)
        ]

    particles_df["rlnImageName"] = image_names
    starfile_dict["particles"] = particles_df
    starfile.write(starfile_dict, filename)

    return
