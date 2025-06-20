"""https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from ...internal import NDArrayLike


LEBDEV_N_PSI_LOOKUP_TABLE = {
    6: 4,  # precision 3
    14: 6,  # precision 5
    26: 8,  # precision 7
    38: 10,  # precision 9
    50: 12,  # precision 11
    74: 14,  # precision 13
    86: 16,  # precision 15
    110: 18,  # precision 17
    146: 20,  # precision 19
    170: 22,  # precision 21
    194: 24,  # precision 23
    230: 26,  # precision 25
    266: 28,  # precision 27
    302: 30,  # precision 29
    350: 32,  # precision 31
    434: 36,  # precision 35
    590: 42,  # precision 41
    770: 48,  # precision 47
    974: 54,  # precision 53
    1202: 60,  # precision 59
    1454: 66,  # precision 65
    1730: 72,  # precision 71
    2030: 78,  # precision 77
    2354: 84,  # precision 83
    2702: 90,  # precision 89
    3074: 96,  # precision 95
    3470: 102,  # precision 101
    3890: 108,  # precision 107
    4334: 114,  # precision 113
    4802: 120,  # precision 119
    5294: 126,  # precision 125
    5810: 132,  # precision 131
}


def parse_lebedev_file(path: str) -> Tuple[NDArrayLike, NDArrayLike, NDArrayLike]:
    """Helper function to parse Lebedev grid files."""
    data = np.loadtxt(path)
    theta = jnp.array(data[:, 0])
    phi = jnp.array(data[:, 1])
    weight = jnp.array(data[:, 2])
    return theta, phi, weight


def build_euler_angle_grid(
    thetas: Float[NDArrayLike, "n"],  # noqa: F821
    phis: Float[NDArrayLike, "n"],  # noqa: F821
    psis: Float[NDArrayLike, "m"],  # noqa: F821
) -> Float[NDArrayLike, "n*m 3"]:  # noqa: F821
    """Build a grid of Euler angles from given theta, phi, and psi angles."""

    n = thetas.shape[0]
    n_psi = psis.shape[0]

    thetas_repeated = jnp.repeat(thetas, n_psi)
    phis_repeated = jnp.repeat(phis, n_psi)
    psis_tiled = jnp.tile(psis, n)

    orientations = jnp.stack([thetas_repeated, phis_repeated, psis_tiled], axis=-1)
    return orientations


def build_lebedev_quadrature(
    fname: str,
) -> Tuple[Float[NDArrayLike, "n*m 3"], Float[NDArrayLike, "n*m"]]:  # noqa: F821
    """Build a Lebedev grid from a file."""

    theta, phi, lebdev_weights = parse_lebedev_file(fname)
    n_psi = LEBDEV_N_PSI_LOOKUP_TABLE[lebdev_weights.shape[0]]
    euler_angles_deg = build_euler_angle_grid(
        theta, phi, jnp.linspace(-180, 180, n_psi, endpoint=False)
    )
    weights_repeated_for_psis = jnp.repeat(lebdev_weights, n_psi)
    return euler_angles_deg, weights_repeated_for_psis / weights_repeated_for_psis.sum()
