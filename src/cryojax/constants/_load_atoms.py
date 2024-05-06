"""
Routines for loading atomic structures.
Large amounts of the code are adapted from the ioSPI package
"""

import importlib.resources as pkg_resources
import os

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int


def _load_peng1996_element_form_factor_param_table():
    """Internal function to load the atomic form factor parameter table."""
    with pkg_resources.as_file(
        pkg_resources.files("cryojax").joinpath("constants")
    ) as path:
        atom_form_factor_params = jnp.load(os.path.join(path, "element_params.npy"))

    return jnp.asarray(atom_form_factor_params)


peng1996_form_factor_param_table = _load_peng1996_element_form_factor_param_table()


def get_form_factor_params_from_table(
    atom_identities: Int[Array, " n_atoms"] | Int[np.ndarray, " n_atoms"],
    form_factor_param_table: (
        Float[Array, "n_params n_elements n_form_factors"]
        | Float[np.ndarray, "n_params n_elements n_form_factors"]
    ),
) -> Float[Array, "n_params n_atoms n_form_factors"]:
    """Gets the parameters for the form factor for each atom in
    `atom_names`.

    **Arguments:**

    - `atom_identitites`:
        Array containing the index of the one-hot encoded atom names.
        By default, Hydrogen is "1", Carbon is "6", Nitrogen is "7", etc.
    - `form_factor_params`:
        Array containing the table of form factors.

    **Returns:**

    The particular form factor parameters stored in `form_factor_param_table` for
    `atom_identities`.
    """
    return jnp.asarray(form_factor_param_table)[:, jnp.asarray(atom_identities)]
