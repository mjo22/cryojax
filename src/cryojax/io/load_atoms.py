"""
Routines for loading atomic structures.
Large amounts of the code are adapted from the ioSPI package
"""

__all__ = [
    "atom_form_factor_params",
    "get_form_factor_params",
]

import numpy as np
import numpy.typing as npt
import os
import jax.numpy as jnp
import importlib.resources as pkg_resources
import pickle
from typing import Iterable, Dict


def _load_element_form_factor_params():
    """
    Internal function to load the atomic form factor parameters.
    """
    with pkg_resources.path("cryojax", "parameters") as path:
        with open(
            os.path.join(path, "atom_form_factor_params.pkl"), "rb"
        ) as f:
            atom_form_factor_params = pickle.load(f)

    return atom_form_factor_params


atom_form_factor_params = _load_element_form_factor_params()


def get_form_factor_params(
    atom_names: Iterable[str],
    form_factor_params: Dict[
        str, Dict[str, npt.NDArray[float]]
    ] = atom_form_factor_params,
):
    """
    Gets the parameters for the form factor for each atom in a list.
    """
    a_vals = np.array([form_factor_params[atom]["a"] for atom in atom_names])

    b_vals = np.array([form_factor_params[atom]["b"] for atom in atom_names])
    return a_vals, b_vals
