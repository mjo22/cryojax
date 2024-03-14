# Welcome to cryoJAX!

`cryojax` is a library for cryo-EM image simulation and analysis. It is built on [`jax`](https://github.com/google/jax).

The core of this package is its ability to model image formation in cryo-EM. The parameters of these models can be estimated for experimental cryo-EM images using standard sampling and optimization libraries in `jax`, such as [`blackjax`](https://github.com/blackjax-devs/blackjax), [`optimistix`](https://github.com/patrick-kidger/optimistix), or [`optax`](https://github.com/google-deepmind/optax). Then, these model parameters can be exported to standard cryo-EM data formats.

Dig a little deeper and you'll find that `cryojax` aims to be a fully extensible modeling language for cryo-EM image formation. It implements a collection of abstract interfaces, which aim to be general enough to support any level of modeling complexity—from simple linear image formation to the most realistic physical models in the field. Best of all, these interfaces are all part of the public API. Users can create their own extensions to `cryojax`, tailored to their specific use-case!

This documentation is currently a work-in-progress. Your patience while we get this project properly documented is much appreciated! Feel free to get in touch on github [issues](https://github.com/mjo22/cryojax/issues) if you have any questions, bug reports, or feature requests.

## Installation

Installing `cryojax` is simple. To start, I recommend creating a new virtual environment. For example, you could do this with `conda`.

```bash
conda create -n cryojax-env -c conda-forge python=3.10
```

Note that `python>=3.10` is required. After creating a new environment, [install JAX](https://github.com/google/jax#installation) with either CPU or GPU support. Then, install `cryojax`. For now, only a source build is supported.

```bash
git clone https://github.com/mjo22/cryojax
cd cryojax
python -m pip install .
```

This will install the remaining dependencies, such as [`equinox`](https://github.com/patrick-kidger/equinox/) for object-oriented model building and `cryojax` core functionality and [`mrcfile`](https://github.com/ccpem/mrcfile) for I/O.

The [`jax-finufft`](https://github.com/dfm/jax-finufft) package is an optional dependency used for non-uniform fast fourier transforms. These are included as an option for computing image projections of real-space voxel-based scattering potential representations. In this case, we recommend first following the `jax_finufft` installation instructions and then installing `cryojax`.

## Quick example

The following is a basic workflow to simulate an image.

First, instantiate the scattering potential representation and its respective method for computing image projections.

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cs
from cryojax.io import read_volume_with_voxel_size_from_mrc

# Instantiate the scattering potential.
filename = "example_scattering_potential.mrc"
real_voxel_grid, voxel_size = read_volume_with_voxel_size_from_mrc(filename)
potential = cs.FourierVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, voxel_size)
# ... now instantiate fourier slice extraction
integrator = cs.FourierSliceExtract(interpolation_order=1)
```

Here, the 3D scattering potential array is read from `filename`. Then, the abstraction of the scattering potential is then loaded in fourier-space into a `FourierVoxelGridPotential`, and the fourier-slice projection theorem is initialized with `FourierSliceExtract`. The scattering potential can be generated with an external program, such as the [cisTEM](https://github.com/timothygrant80/cisTEM) simulate tool.

We can now instantiate the representation of a biological specimen, which also includes a pose.

```python
# First instantiate the pose. Angles are given in degrees
pose = cs.EulerAnglePose(
    offset_x_in_angstroms=5.0,
    offset_y_in_angstroms=-3.0,
    view_phi=20.0,
    view_theta=80.0,
    view_psi=-10.0,
)
# ... now, build the biological specimen
specimen = cs.Specimen(potential, integrator, pose)
```

Next, build the model for the electron microscope. Here, we simply include a model for the CTF in the weak-phase approximation (linear image formation theory).

```python
from cryojax.image import operators as op

# First, initialize the CTF and its optics model
ctf = cs.CTF(
    defocus_u_in_angstroms=10000.0,
    defocus_v_in_angstroms=9800.0,
    astigmatism_angle=10.0,
    voltage_in_kilovolts=300.0,
    amplitude_contrast_ratio=0.1)
optics = cs.WeakPhaseOptics(ctf, envelope=op.FourierGaussian(b_factor=5.0))  # b_factor is given in Angstroms^2
# ... these are stored in the Instrument
instrument = cs.Instrument(optics)
```

The `CTF` has all parameters used in CTFFIND4, which take their default values if not
explicitly configured here. Finally, we can instantiate the `ImagePipeline` and simulate an image.

```python
# Instantiate the image configuration
config = cs.ImageConfig(shape=(320, 320), pixel_size=voxel_size)
# Build the image formation model
pipeline = cs.ImagePipeline(config, specimen, instrument)
# ... simulate an image and return a normalized image in real-space
image = pipeline.render(get_real=True, normalize=True)
```

## Next steps

There are many modeling features in `cryojax` to learn about. To learn more, see the Examples section of the documentation.

## Acknowledgements

- `cryojax` has been greatly informed by the open-source cryo-EM softwares [`cisTEM`](https://github.com/timothygrant80/cisTEM) and [`BioEM`](https://github.com/bio-phys/BioEM).
- `cryojax` relies heavily on and has taken great inspiration from [`equinox`](https://github.com/patrick-kidger/equinox/). We think that `equinox` has great design principles and highly recommend learning about it to fully make use of the power of `jax`.
