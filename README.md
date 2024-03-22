<h1 align='center'>cryoJAX</h1>

![Tests](https://github.com/mjo22/cryojax/actions/workflows/testing.yml/badge.svg)
![Lint](https://github.com/mjo22/cryojax/actions/workflows/black.yml/badge.svg)

cryoJAX is a library for cryo-EM image simulation and analysis. It is built on [`jax`](https://github.com/google/jax).

## Summary

The core of this package is its ability to model image formation in cryo-EM. The parameters of these models can be estimated for experimental cryo-EM images using standard sampling and optimization libraries in `jax`, such as [`blackjax`](https://github.com/blackjax-devs/blackjax), [`optimistix`](https://github.com/patrick-kidger/optimistix), or [`optax`](https://github.com/google-deepmind/optax). Then, these model parameters can be exported to standard cryo-EM data formats.

Dig a little deeper and you'll find that `cryojax` aims to be a fully extensible modeling language for cryo-EM image formation. It implements a collection of abstract interfaces, which aim to be general enough to support any level of modeling complexity—from simple linear image formation to the most realistic physical models in the field. Best of all, these interfaces are all part of the public API. Users can create their own extensions to `cryojax`, tailored to their specific use-case!

## Documentation

See the documentation at [https://mjo22.github.io/cryojax/](https://mjo22.github.io/cryojax/). It is a work-in-progress, so thank you for your patience!

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

## Simulating an image

The following is a basic workflow to simulate an image.

First, instantiate the scattering potential representation and its respective method for computing image projections.

```python
import jax
import jax.numpy as jnp
import cryojax.simulator as cs
from cryojax.io import read_array_with_spacing_from_mrc

# Instantiate the scattering potential.
filename = "example_scattering_potential.mrc"
real_voxel_grid, voxel_size = read_array_with_spacing_from_mrc(filename)
potential = cs.FourierVoxelGridPotential.from_real_voxel_grid(real_voxel_grid, voxel_size)
# ... now instantiate fourier slice extraction
integrator = cs.FourierSliceExtract(interpolation_order=1)
```

Here, the 3D scattering potential array is read from `filename`. Then, the abstraction of the scattering potential is then loaded in fourier-space into a `FourierVoxelGridPotential`, and the fourier-slice projection theorem is initialized with `FourierSliceExtract`. The scattering potential can be generated with an external program, such as the [cisTEM](https://github.com/timothygrant80/cisTEM) simulate tool.

We can now instantiate the representation of a biological specimen, which also includes a pose.

```python
# First instantiate the pose. Here, angles are given in degrees
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

The `CTF` has parameters used in CTFFIND4, which take their default values if not
explicitly configured here. Finally, we can instantiate the `ImagePipeline` and simulate an image.

```python
# Instantiate the image configuration
config = cs.ImageConfig(shape=(320, 320), pixel_size=voxel_size)
# Build the image formation model
pipeline = cs.ImagePipeline(config, specimen, instrument)
# ... simulate an image and return in real-space.
image_without_noise = pipeline.render(get_real=True)
```

`cryojax` also defines a library of distributions from which to sample the data. These distributions define the stochastic model from which images are drawn. For example, instantiate an `IndependentFourierGaussian` distribution and either sample from it or compute its log-likelihood.

```python
from cryojax.image import rfftn
from cryojax.inference import distributions as dist
from cryojax.image import operators as op

# Passing the ImagePipeline and a variance function, instantiate the distribution
distribution = dist.IndependentFourierGaussian(pipeline, variance=op.Constant(1.0))
# ... then, either simulate an image from this distribution
key = jax.random.PRNGKey(seed=0)
image_with_noise = distribution.sample(key)
# ... or compute the likelihood
observed = rfftn(...)  # for this example, read in observed data and take FFT
log_likelihood = distribution.log_likelihood(observed)
```

For more advanced image simulation examples and to understand the many features in this library, see the [documentation](https://mjo22.github.io/cryojax/).

## Acknowledgements

- `cryojax` has been greatly informed by the open-source cryo-EM softwares [`cisTEM`](https://github.com/timothygrant80/cisTEM) and [`BioEM`](https://github.com/bio-phys/BioEM).
- `cryojax` relies heavily on and has taken great inspiration from [`equinox`](https://github.com/patrick-kidger/equinox/). We think that `equinox` has great design principles and highly recommend learning about it to fully make use of the power of `jax`.
