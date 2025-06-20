"""https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from ...internal import NDArrayLike


lebedev_n_psi_lookup = {
    6: 2,  # precision 3
    14: 2,  # precision 5
    26: 2,  # precision 7
    38: 2,  # precision 9
    50: 3,  # precision 11
    74: 3,  # precision 13
    86: 4,  # precision 15
    110: 4,  # precision 17
    146: 5,  # precision 19
    170: 6,  # precision 21
    194: 6,  # precision 23
    230: 6,  # precision 25
    266: 7,  # precision 27
    302: 8,  # precision 29
    350: 8,  # precision 31
    434: 9,  # precision 35
    590: 10,  # precision 41
    770: 12,  # precision 47
    974: 14,  # precision 53
    1202: 15,  # precision 59
    1454: 16,  # precision 65
    1730: 18,  # precision 71
    2030: 20,  # precision 77
    2354: 22,  # precision 83
    2702: 24,  # precision 89
    3074: 26,  # precision 95
    3470: 28,  # precision 101
    3890: 30,  # precision 107
    4334: 32,  # precision 113
    4802: 34,  # precision 119
    5294: 36,  # precision 125
    5810: 38,  # precision 131
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
    n_psi = lebedev_n_psi_lookup[lebdev_weights.shape[0]]
    euler_angles_deg = build_euler_angle_grid(
        theta, phi, jnp.linspace(-180, 180, n_psi, endpoint=False)
    )
    weights_repeated_for_psis = jnp.repeat(lebdev_weights, n_psi)
    return euler_angles_deg, weights_repeated_for_psis / weights_repeated_for_psis.sum()
