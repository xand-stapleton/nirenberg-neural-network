import numpy as np
import tensorflow as tf


def r_to_R(r: float) -> float:
    """
    Convert a disc radius r (between 0 and 1) to a stereographic radius R.

    This conversion is used in the modified stereographic projection that maps
    the unit disc to the sphere. The formula is: R = 2r / (1 - r^2).

    Args:
        r: disc radius in [0, 1)

    Returns:
        float: stereographic radius R
    """
    return 2.0 * r / (1.0 - r**2)


def xyz_to_patch_xy(xyz: tf.Tensor, patch_idx: int) -> tf.Tensor:
    """
    Standard stereographic projection (sphere -> disc).

    Args:
        xyz: (num_samples, 3)
            embedded sphere coordinates
        patch_idx: 0 (northern hemisphere, project from south pole)
                  1 (southern hemisphere, project from north pole)

    Returns:
        tf.Tensor: (num_samples, 2)
            chart coordinates

    Notes:
        - patch_idx=0: Projects northern hemisphere (z > 0) from south pole,
          mapping it to interior of unit disc (r < 1)
        - patch_idx=1: Projects southern hemisphere (z < 0) from north pole,
          mapping it to interior of unit disc (r < 1)
        - For readability, we do not guard against numerical instability. It is assumed
          that the xyz coords are sufficiently far from the projection pole

    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if patch_idx == 0:
        # Project from south pole (for northern hemisphere, z > 0)
        denom = 1.0 + z
    elif patch_idx == 1:
        # Project from north pole (for southern hemisphere, z < 0)
        denom = 1.0 - z
    else:
        raise ValueError

    # Standard stereographic projection
    u, v = x / denom, y / denom

    return tf.stack([u, v], axis=-1)


def patch_xy_to_xyz(uv: tf.Tensor, patch_idx: int) -> tf.Tensor:
    """
    Inverse standard stereographic projection (disc -> sphere).

    Args:
        uv: (num_samples, 2)
            chart coordinates
        patch_idx: 0 (northern hemisphere)
                  1 (southern hemisphere)

    Returns:
        tf.Tensor: (num_samples, 3)
            embedded sphere coordinates

    Notes:
        - patch_idx=0: Inverse for northern hemisphere (gives z > 0)
        - patch_idx=1: Inverse for southern hemisphere (gives z < 0)
        - For readability, we do not guard against numerical instability. It is assumed
          that the uv coords are sufficiently far from the disc boundary

    """
    x1, x2 = uv[:, 0], uv[:, 1]
    r2 = x1**2 + x2**2

    denom = 1.0 + r2
    X = 2.0 * x1 / denom
    Y = 2.0 * x2 / denom
    Z = (1.0 - r2) / denom
    
    if patch_idx == 1:  # Southern hemisphere flips
        Z = -Z

    return tf.stack([X, Y, Z], axis=-1)


def analytic_round_metric(
    coords: tf.Tensor,
    identity: bool = False,
    eps: float = 1e-12,
) -> tf.Tensor:
    """
    Compute the round metric (standard metric on the unit sphere) in patch coordinates.

    The round metric on the sphere, when expressed in standard stereographic coordinates,
    has the form: g = 4/(1+r²)² I, where r² = ||x||² and I is the identity matrix.
    This metric has constant scalar curvature 2.

    Args:
        coords: (N, 2)
            patch coordinates (u, v) in the unit disc
        identity: if True, return identity metric instead
        eps: small epsilon for numerical stability when clamping r^2

    Returns:
        tf.Tensor: (N, 2, 2)
            metric tensor at each coordinate point
    """
    N, D = tf.shape(coords)[0], tf.shape(coords)[1]

    if identity:
        return tf.eye(D, batch_shape=[N], dtype=coords.dtype)

    r2 = tf.reduce_sum(tf.square(coords), axis=-1, keepdims=True)

    # Standard stereographic metric: g = 4/(1+r²)² I
    conformal_factor = 4.0 / tf.square(1.0 + r2)
    I = tf.eye(D, batch_shape=[N], dtype=coords.dtype)

    return conformal_factor[..., None] * I
