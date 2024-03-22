# Contributor Guide

Contributions to this repository are welcome and greatly appreciated! I would love
for this package to grow and be supported by a larger community.

## Design principles

`cryojax` is built on [equinox](https://docs.kidger.site/equinox/). In short, `equinox` provides an object-oriented interface to writing parameterized functions in `jax`. The core object of these parameterized functions is called a [Module](https://docs.kidger.site/equinox/api/module/module/) (yes, this takes inspiration from pytorch). `equinox` ships with features to interact with these `Module`s, and more generally with [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) in `jax`. One of the most useful of these features, not found in `jax` itself, is a means of performing out-of-place updates on pytrees through `equinox.tree_at`.

Equinox also provides a recommended pattern for writing `Module`s: https://docs.kidger.site/equinox/pattern/. We think this is a good template for code readability, so `cryojax` tries to adhere to these principles as much as possible.

## Running tests and building the documentation

Both the tests and documentation use files stored remotely with git [LFS](https://git-lfs.com/). You will need to install `git lfs` locally in order to get these files.

## How to report a bug

Report bugs on the [Issue Tracker](https://github.com/mjo22/cryojax/issues).

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case, and/or steps to
reproduce the issue. In particular, please include a [Minimal, Reproducible
Example](https://stackoverflow.com/help/minimal-reproducible-example).

## How to request a feature

Feel free to request features on the [Issue
Tracker](https://github.com/mjo22/cryojax/issues).

## How to submit changes

Open a [Pull Request](https://github.com/mjo22/cryojax/pulls).
