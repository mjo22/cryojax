# 2D Template Matching in JAX
An implementation of 2D template matching (2DTM) in cryo electron microscopy micrographs built on [JAX](https://github.com/google/jax).

## Installation

Installing `jax-2dtm` is currently a bit more complicated than it should be because it relies on the library [tensorflow-nufft](https://github.com/mrphys/tensorflow-nufft) through the `jax` experimental feature [jax2tf](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md). Getting [tensorflow](https://github.com/tensorflow/tensorflow) to install besides `jax` is a bit of a pain. Once [jax-finufft](https://github.com/dfm/jax-finufft) adds GPU support, the `tensorflow` dependency will be replaced. In the following instructions, please install `tensorflow` or `jax` with the appropriate configuration with your workflow. Namely, install both with GPU support if you would like, or not!

To start, I recommend creating a new virtual environment. For example, you could do this with `conda`:

```bash
conda create -n jax-2dtm -c conda-forge python=3.10
```

I recommend using `python=3.10` because of recent features and changing best practices in type checking, but really all that is necessary is support for python [dataclasses](https://docs.python.org/3/library/dataclasses.html) (`python>=3.7`). Custom dataclasses that are safe to pass to `jax` are heavily used in this library!

Then, modifiy the [tensorflow installation](https://www.tensorflow.org/install/pip) instructions to install version 2.11.x. Be careful to not install the latest version! As I'm writing this, `tensorflow-nufft` supports up to version 2.11.x. This version only suppports up to `3.7<=python<=3.10`, so make sure you're not using `python>=3.11`.

Finally, [install tensorflow-nufft](https://mrphys.github.io/tensorflow-nufft/guide/start/) with

```bash
pip install tensorflow-nufft==0.12.0
```

Next, [install JAX](https://github.com/google/jax#installation). I have found it easier to first install `tensorflow` and then `jax`, but it may be possible to reverse these instructions.

Finally, install `jax-2dtm`. For now, only a source build is supported.

```bash
git clone https://github.com/mjo22/jax-2dtm
cd jax-2dtm
python -m pip install .
```

This will install the remaining dependencies, such as [jaxlie](https://github.com/brentyi/jaxlie) for coordinate rotations and translations, [mrcfile](https://github.com/ccpem/mrcfile) for IO, and [dataclasses-json](https://github.com/lidatong/dataclasses-json) for serialization.

## Features

Thanks to `jax`, imaging models in `jax-2dtm` support functional transformations, such as automatic differentiation with `grad` and paralellization with `vmap` and `pmap`. Models also support GPU/TPU acceleration. Until GPU support for `jax-finufft` is added, `jit` compilation will not be supported because `tensorflow-nufft` does not compile.

## Usage

Please note that this library is currently experimental and the API is subject to change!

## Similar libraries

- [cisTEM](https://github.com/timothygrant80/cisTEM): A software to process cryo-EM images of macromolecular complexes and obtain high-resolution 3D reconstructions from them. The recent experimental release of `cisTEM` has implemented a highly successful 2DTM program.

- [BioEM](https://github.com/bio-phys/BioEM): Bayesian inference of Electron Microscopy. This codebase calculates the posterior probability of a structural model given multiple experimental EM images.
