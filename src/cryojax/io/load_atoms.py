"""
Routines for loading atomic structures.
Large amounts of the code are adapted from the ioSPI package
"""

__all__ = [
    "default_form_factor_params",
    "get_form_factor_params",
]

import numpy as np
import jax
import jax.numpy as jnp
import os
import importlib.resources as pkg_resources
from cryojax.typing import IntCloud
from jaxtyping import Array, Float
from functools import partial


def _load_element_form_factor_params():
    """
    Internal function to load the atomic form factor parameters.
    """
    with pkg_resources.path("cryojax", "parameters") as path:
        atom_form_factor_params = np.load(
            os.path.join(path, "element_params.npy")
        )

    return jnp.array(atom_form_factor_params)


default_form_factor_params = _load_element_form_factor_params()


@partial(jax.jit, static_argnums=(1,))
def get_form_factor_params(
    atom_names: IntCloud,
    form_factor_params: Float[Array, "2 N k"] = None,
):
    """
    Gets the parameters for the form factor for each atom in atom_names.

    Parameters
    ----------
    atom_names : npt.NDArray[int]
        Array containing the index of the one-hot encoded atom names.
        By default, Hydrogen is "1", Carbon is "6", Nitrogen is "7", etc.
    a_params : npt.NDArray[float], optional
        Array containing the strength of the Gaussian for each form factor
    b_params : npt.NDArray[float], optional
        Array containing the scaling of the Gaussian for each form factor
    """

    if form_factor_params is None:
        data = default_form_factor_params[:, atom_names]
        return data[0], data[1]
    else:
        data = form_factor_params[:, atom_names]
        return data[0], data[1]
