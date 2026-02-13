from typing import Callable

import numpy as np
import tensorflow as tf


def integrate_monte_carlo(
    f: Callable,
    num_samples: int = 100_000,
    dtype: tf.DType = tf.float64,
) -> float:
    """
    Monte Carlo estimate of the integral of a function f(θ, φ) over the sphere.

    Args:
        f: integrand, must be a function of θ and φ
        num_samples: number of Monte Carlo samples
        dtype: data type

    Returns:
        float: integral of f over the sphere

    """
    # Uniformly sample z ∈ [-1, 1]
    z = tf.random.uniform(
        shape=[num_samples],
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
    )
    # Uniformly sample φ ∈ [0, 2π)
    phi = tf.random.uniform(
        shape=[num_samples],
        minval=0.0,
        maxval=2.0 * np.pi,
        dtype=dtype,
    )
    # Convert z -> θ
    theta = tf.acos(z)
    # Add trailing dimension for compatibility with f shape expectations
    theta = theta[..., None]
    phi = phi[..., None]
    # Evaluate function
    values = f(theta, phi)
    values = tf.squeeze(values, axis=-1)
    # Monte Carlo estimate: ∫ f dΩ ≈ 4π * mean(f)
    result = tf.cast(4.0 * np.pi, dtype) * tf.reduce_mean(values)
    return float(result.numpy())


def integrate_gauss_legendre(
    f: Callable,
    num_theta: int = 200,
    num_phi: int = 500,
    dtype: tf.DType = tf.float64,
) -> float:
    """
    Gauss-Legendre quadrature estimate of the integral of a function f(θ, φ) over the
    sphere.

    Args:
        f: integrand, must be a function of θ and φ
        num_theta: number of colatitude angles
        num_phi: number of azimuthal angles
        dtype: data type

    Returns:
        float: integral of f over the sphere

    """
    # Gauss-Legendre quadrature scheme on z ∈ [-1, 1] (where z = cos θ)
    z_nodes, z_weights = np.polynomial.legendre.leggauss(num_theta)
    z_nodes = tf.constant(z_nodes, dtype=dtype)
    z_weights = tf.constant(z_weights, dtype=dtype)
    # Uniform trapezoidal quadrature scheme on φ ∈ [0, 2π)
    dphi = tf.cast(2.0 * np.pi / num_phi, dtype)
    phi_nodes = dphi * tf.cast(tf.range(num_phi), dtype)
    # Build mesh (z, φ)
    z_grid, phi_grid = tf.meshgrid(z_nodes, phi_nodes, indexing="ij")
    # Convert z -> θ
    theta_grid = tf.acos(z_grid)
    # Add trailing dimension for compatibility with f shape expectations
    theta_grid = theta_grid[..., None]
    phi_grid = phi_grid[..., None]
    # Evaluate function
    values = tf.squeeze(f(theta_grid, phi_grid), axis=-1)
    # Gauss-Legendre quadrature ⊗ trapezoidal quadrature
    result = tf.reduce_sum((tf.reduce_sum(values, axis=1) * dphi) * z_weights)
    return float(result.numpy())
