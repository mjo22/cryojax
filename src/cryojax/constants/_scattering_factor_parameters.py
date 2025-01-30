"""
Routines for loading atomic structures.
Large amounts of the code are adapted from the ioSPI package
"""

import importlib.resources as pkg_resources
import os
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def get_tabulated_scattering_factor_parameters(
    atom_identities: Int[Array, " n_atoms"] | Int[np.ndarray, " n_atoms"],
    scattering_factor_parameter_table: Optional[
        Float[Array, "n_params n_elements n_scattering_factors"]
        | Float[np.ndarray, "n_params n_elements n_scattering_factors"]
    ] = None,
) -> Float[Array, "n_params n_atoms n_scattering_factors"]:
    """Gets the parameters for the scattering factor for each atom in
    `atom_identities`.

    **Arguments:**

    - `atom_identitites`:
        Array containing the index of the one-hot encoded atom names.
        By default, Hydrogen is "1", Carbon is "6", Nitrogen is "7", etc.
    - `scattering_factor_parameter_table`:
        Array containing the table of scattering factors. By default, this
        is the tabulation from "Robust Parameterization of Elastic and
        Absorptive Electron Atomic Scattering Factors" by Peng et al. (1996),
        given by [`read_peng_element_scattering_factor_parameter_table`](https://mjo22.github.io/cryojax/api/constants/scattering_factor_parameters/#cryojax.constants.read_peng_element_scattering_factor_parameter_table).

    **Returns:**

    The particular scattering factor parameters stored in
    `scattering_factor_parameter_table` for `atom_identities`.
    """  # noqa: #E501
    if scattering_factor_parameter_table is None:
        scattering_factor_parameter_table = (
            read_peng_element_scattering_factor_parameter_table()
        )
    return jnp.asarray(scattering_factor_parameter_table)[:, jnp.asarray(atom_identities)]


def read_peng_element_scattering_factor_parameter_table() -> (
    Float[np.ndarray, "2 n_elements 5"]
):
    r"""Function to load the atomic scattering factor parameter
    table from "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996).

    **Returns:**

    The parameter table for parameters $\{a_i\}_{i = 1}^5$ and $\{b_i\}_{i = 1}^5$
    for each atom, described in the above reference.
    """
    with pkg_resources.as_file(
        pkg_resources.files("cryojax").joinpath("constants")
    ) as path:
        atom_scattering_factor_params = jnp.load(
            os.path.join(path, "peng1996_element_params.npy")
        )

    return np.asarray(atom_scattering_factor_params)
