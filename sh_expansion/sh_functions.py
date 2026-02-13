"""
Shared functions for spherical harmonics expansion scripts.

This module contains common functionality used by:
- run_sh_direct.py: Direct optimization of SH coefficients
- run_sh_model.py: Fitting SH coefficients to model predictions
- load_sh.py: Loading and evaluating saved coefficients
"""

import math
from typing import Tuple

import numpy as np
import tensorflow as tf

from geometry.ball import analytic_round_metric, patch_xy_to_xyz


# ============================================================================
# Common defaults and constants
# ============================================================================

DEFAULT_L_MAX = 4
DEFAULT_DTYPE = "float64"
DEFAULT_SEED = 42

# Optimization defaults for coefficient optimization
DEFAULT_MODEL_CHECKPOINT = "."
DEFAULT_NUM_PATCHES = 2
DEFAULT_NUM_SAMPLES = 20000
DEFAULT_EPOCHS_MODEL = 500
DEFAULT_LR = 0.1
DEFAULT_LR_END = 0.0
EARLY_STOPPING_PATIENCE = 50
REGULARIZATION_WEIGHT = 0.000000001
EPSILON_COEFF_DISPLAY = 1e-2


# ============================================================================
# Helper functions for argument parsing
# ============================================================================

def add_common_args(parser, method: str = "direct"):
    """
    Add common command-line arguments to an argparse parser.
    
    Args:
        parser: argparse.ArgumentParser instance
        method: "direct" or "model" to customize defaults
        
    Common arguments added:
        --l_max: Maximum degree of spherical harmonics expansion
        --dtype: TensorFlow data type
        --seed: Random seed for reproducibility
    """
    parser.add_argument(
        "--l_max",
        type=int,
        default=DEFAULT_L_MAX,
        help=f"Maximum degree of spherical harmonics expansion (default: {DEFAULT_L_MAX})"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["float32", "float64"],
        help=f"TensorFlow data type (default: {DEFAULT_DTYPE})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    
    return parser


def real_spherical_harmonic_tf(
    l: int,
    m: int,
    theta: tf.Tensor,
    phi: tf.Tensor,
) -> tf.Tensor:
    """
    Compute real-valued spherical harmonic Y_l^m(θ, φ) using TensorFlow.
    
    This matches the implementation in data/prescribers.py (SphericalHarmonics class).
    
    Args:
        l: non-negative degree
        m: order with |m| <= l
        theta: (...,) colatitudes in radians
        phi: (...,) longitudes in radians
        
    Returns:
        tf.Tensor: (...,) values of real spherical harmonic
    """
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")
    
    # Associated Legendre polynomial P_l^|m|
    x = tf.cos(theta)
    m_abs = abs(m)
    
    # Compute P_m^m using double factorial
    one_minus_x2 = 1.0 - x**2
    
    if m_abs == 0:
        Pmm = tf.ones_like(x)
    else:
        df = math.factorial(2 * m_abs) // (2**m_abs * math.factorial(m_abs))
        cs = (-1) ** m_abs
        Pmm = cs * df * tf.pow(one_minus_x2, tf.cast(0.5 * m_abs, x.dtype))
    
    if l == m_abs:
        Pnm = Pmm
    elif l == m_abs + 1:
        Pnm = x * (2.0 * m_abs + 1.0) * Pmm
    else:
        # Three-term recurrence
        Pnm2 = Pmm
        Pnm1 = x * (2.0 * m_abs + 1.0) * Pmm
        for n in range(m_abs + 2, l + 1):
            term1 = (2.0 * n - 1.0) * x * Pnm1
            term2 = (n + m_abs - 1.0) * Pnm2
            Pn = (term1 - term2) / (n - m_abs)
            Pnm2, Pnm1 = Pnm1, Pn
        Pnm = Pnm1
    
    # Normalization factor
    prefactor = (2.0 * l + 1.0) / (4.0 * np.pi)
    a = math.lgamma(l - abs(m) + 1)
    b = math.lgamma(l + abs(m) + 1)
    log_ratio = a - b
    Nlm = tf.sqrt(tf.cast(prefactor * np.exp(log_ratio), theta.dtype))
    
    # Branch on m
    if m == 0:
        return Nlm * Pnm
    elif m > 0:
        cos_mphi = tf.cos(tf.cast(m, phi.dtype) * phi)
        return tf.cast(np.sqrt(2.0), theta.dtype) * Nlm * Pnm * cos_mphi
    else:  # m < 0
        sin_mphi = tf.sin(tf.cast(abs(m), phi.dtype) * phi)
        return tf.cast(np.sqrt(2.0), theta.dtype) * Nlm * Pnm * sin_mphi


def xyz_to_spherical(xyz: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convert Cartesian coordinates to spherical coordinates (θ, φ).
    
    Args:
        xyz: (..., 3) Cartesian coordinates on the unit sphere
        
    Returns:
        theta: (...,) colatitudes in [0, π]
        phi: (...,) longitudes in [0, 2π)
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    # Colatitude: θ = arccos(z)
    theta = tf.acos(tf.clip_by_value(z, -1.0, 1.0))
    
    # Longitude: φ = atan2(y, x), mapped to [0, 2π)
    phi = tf.atan2(y, x)
    phi = tf.where(phi < 0, phi + 2 * np.pi, phi)
    
    return theta, phi


@tf.function(reduce_retracing=True)
def compute_laplace_beltrami_tf(
    u_model,
    patch_coords: tf.Tensor,
    patch_idx: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute u and its Laplace-Beltrami operator Δ_g0 u using autodifferentiation.
    
    This matches the computation in network/global_conformal_model.py but uses TensorFlow.
    
    Args:
        u_model: callable that takes (theta, phi) and returns u
        patch_coords: (N, 2) patch coordinates
        patch_idx: 0 (north) or 1 (south)
        
    Returns:
        u: (N,) conformal factor values
        delta_u: (N,) Laplace-Beltrami of u
    """
    # Divergence of weighted gradient
    # Note: Tapes must be nested for second-order derivatives to work correctly.
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(patch_coords)
        # Gradient of u w.r.t. patch coords (computed inside t2's context)
        with tf.GradientTape() as t1:
            t1.watch(patch_coords)
            xyz = patch_xy_to_xyz(patch_coords, patch_idx)
            theta, phi = xyz_to_spherical(xyz)
            u = u_model(theta, phi)
        grad_u = t1.gradient(u, patch_coords)
        # Evaluate (round) metric at patch coords
        g0 = analytic_round_metric(patch_coords)
        # Compute determinant and square root
        det_g0 = tf.linalg.det(g0)
        sqrt_det_g0 = tf.sqrt(det_g0)
        # Compute metric inverse
        g0_inv = tf.linalg.inv(g0)
        # Raise index: grad_u_raised^i = g0^{ij} ∂_j u
        grad_u_raised = tf.einsum("nij,nj->ni", g0_inv, grad_u)
        # Weight by √|g0|: weighted_grad_u^i = √|g0| * grad_u_raised^i
        weighted_grad_u = sqrt_det_g0[..., None] * grad_u_raised
        
    # Final divergence: div = ∂_i (weighted_grad_u^i)
    J = t2.batch_jacobian(weighted_grad_u, patch_coords)
    div_weighted_grad = J[:, 0, 0] + J[:, 1, 1]

    # Final Laplace–Beltrami: Δu = (1/√|g0|) * div
    delta_u = div_weighted_grad / sqrt_det_g0
    
    # Clean up persistent tape
    del t2
    
    return u, delta_u


def compute_scalar_curvature(
    u: tf.Tensor,
    delta_u: tf.Tensor,
) -> tf.Tensor:
    """
    Compute scalar curvature R_g from conformal factor u and its Laplacian.
    
    For conformal metric g = e^(2u) * g0 where g0 has constant curvature 2:
        R_g = e^(-2u) * (2 - 2*Δ_g0 u)
    
    This matches the formula in loss/losses.py (ScalarLoss).
    
    Args:
        u: (N,) conformal factor
        delta_u: (N,) Laplace-Beltrami of u
        
    Returns:
        tf.Tensor: (N,) scalar curvature values
    """
    return tf.exp(-2.0 * u) * (2.0 - 2.0 * delta_u)


def generate_samples(
    num_samples: int,
    num_patches: int,
    radial_offset: float,
    dtype: tf.DType,
) -> tf.Tensor:
    """
    Generate uniformly distributed samples on the sphere in patch coordinates.
    
    This is a simplified version that generates uniform samples on the sphere.
    
    Args:
        num_samples: number of samples per patch
        num_patches: 1 or 2 (north only, or both hemispheres)
        radial_offset: angular offset (currently not fully implemented)
        dtype: tf data type
        
    Returns:
        tf.Tensor: (num_patches, num_samples, 2) patch coordinates
    """
    samples = []
    
    for patch_idx in range(num_patches):
        # Generate uniform samples on sphere using spherical coordinates
        # Uniform in cos(theta) ensures uniform distribution on sphere
        z = tf.random.uniform([num_samples], minval=-1.0, maxval=1.0, dtype=dtype)
        phi = tf.random.uniform([num_samples], minval=0.0, maxval=2.0 * np.pi, dtype=dtype)
        
        # Convert to Cartesian
        theta = tf.acos(z)
        x = tf.sin(theta) * tf.cos(phi)
        y = tf.sin(theta) * tf.sin(phi)
        z = tf.cos(theta)
        
        # For patch 0 (north), keep northern hemisphere
        # For patch 1 (south), flip to southern hemisphere
        if patch_idx == 1:
            z = -z
        
        xyz = tf.stack([x, y, z], axis=-1)
        
        # Convert to patch coordinates using forward projection
        # For northern patch (patch_idx=0): project from south pole
        # For southern patch (patch_idx=1): project from north pole
        if patch_idx == 0:
            denom = 1.0 + z
        else:
            denom = 1.0 - z
        
        # Add small epsilon to avoid division by zero at poles
        denom = tf.maximum(denom, 1e-6)
        u = x / denom
        v = y / denom
        
        patch_coords = tf.stack([u, v], axis=-1)
        samples.append(patch_coords)
    
    return tf.stack(samples, axis=0)  # (num_patches, num_samples, 2)


def evaluate_prescriber_tf(
    prescriber,
    patch_coords: tf.Tensor,
    patch_idx: int,
) -> tf.Tensor:
    """
    Evaluate the prescribed scalar curvature at given patch coordinates.
    
    Uses the TensorFlow prescriber directly (no framework conversion needed!).
    
    Args:
        prescriber: prescriber object from data/prescribers.py
        patch_coords: (N, 2) patch coordinates (TensorFlow)
        patch_idx: patch index
        
    Returns:
        tf.Tensor: (N,) prescribed scalar curvature values
    """
    # Convert to xyz
    xyz = patch_xy_to_xyz(patch_coords, patch_idx)
    
    # Convert to spherical coordinates
    theta, phi = xyz_to_spherical(xyz)
    
    # Reshape for prescriber (expects (..., 1) shape)
    theta = tf.reshape(theta, [-1, 1])
    phi = tf.reshape(phi, [-1, 1])
    
    # Evaluate prescriber (TensorFlow)
    prescribed_R = prescriber.forward(theta, phi)
    
    # Flatten to (N,)
    return tf.reshape(prescribed_R, [-1])


def evaluate_u(
    coefficients: np.ndarray,
    lm_pairs: np.ndarray,
    theta: tf.Tensor,
    phi: tf.Tensor
) -> tf.Tensor:
    """
    Evaluate the conformal factor u at given spherical coordinates.
    
    Uses the ansatz: u(θ, φ) = Σ (c_l / (l(l+1))) * Y_l^m(θ, φ)
    to match the normalization in data/prescribers.py SpectralPair.
    
    Args:
        coefficients: (N,) array of spherical harmonics coefficients
        lm_pairs: (N, 2) array of (l, m) indices
        theta: (...,) colatitudes in radians
        phi: (...,) longitudes in radians
        
    Returns:
        tf.Tensor: (...,) values of u
    """
    coeffs_tf = tf.constant(coefficients, dtype=theta.dtype)
    
    result = tf.zeros_like(theta)
    
    for idx, (l, m) in enumerate(lm_pairs):
        coeff = coeffs_tf[idx]
        l_val = tf.cast(l, theta.dtype)
        # Apply normalization: c_l / (l(l+1))
        normalized_coeff = coeff / (l_val * (l_val + 1.0))
        Ylm = real_spherical_harmonic_tf(int(l), int(m), theta, phi)
        result = result + normalized_coeff * Ylm
    
    return result


class SphericalHarmonicsExpansion:
    """
    Represents the conformal factor u as a finite spherical harmonics expansion:
    
        u(θ, φ) = Σ_{l=1}^{l_max} Σ_{m=-l}^{l} (c_{l,m} / (l(l+1))) * Y_l^m(θ, φ)
    
    Note: We start from l=1 (not l=0) to match SpectralPair prescribers, which
    require l >= 1 due to the division by l(l+1) in their conformal factor formula.
    The 1/(l(l+1)) normalization factor matches the ansatz used in data/prescribers.py.
    
    The coefficients c_{l,m} are learnable parameters.
    
    Args:
        max_degree: maximum degree l_max of the expansion
        dtype: tf data type (default: tf.float64)
    """
    
    def __init__(self, max_degree: int, dtype: tf.DType = tf.float64):
        self.max_degree = max_degree
        self.dtype = dtype
        
        # Create list of (l, m) pairs for all spherical harmonics up to max_degree
        # Starting from l=1 (not l=0) to be consistent with SpectralPair prescribers
        self.lm_pairs = []
        for l in range(1, max_degree + 1):  # Changed from range(max_degree + 1)
            for m in range(-l, l + 1):
                self.lm_pairs.append((l, m))
        
        # Total number of coefficients
        num_coeffs = len(self.lm_pairs)
        
        # Initialize coefficients (learnable parameters)
        # Start with small random values near zero
        self.coefficients = tf.Variable(
            tf.random.normal([num_coeffs], dtype=dtype) * 0.01,
            trainable=True,
            name='sh_coefficients'
        )
        
        print(f"Initialized {num_coeffs} spherical harmonic coefficients for l_max={max_degree}")
    
    def __call__(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        """
        Evaluate u at given spherical coordinates.
        
        Uses the ansatz: u(θ, φ) = Σ (c_l / (l(l+1))) * Y_l^m(θ, φ)
        
        Args:
            theta: (...,) colatitudes in radians
            phi: (...,) longitudes in radians
            
        Returns:
            tf.Tensor: (...,) values of u
        """
        result = tf.zeros_like(theta, dtype=self.dtype)
        
        for idx, (l, m) in enumerate(self.lm_pairs):
            coeff = self.coefficients[idx]
            l_val = tf.cast(l, self.dtype)
            # Apply normalization: c_l / (l(l+1))
            normalized_coeff = coeff / (l_val * (l_val + 1.0))
            Ylm = real_spherical_harmonic_tf(l, m, theta, phi)
            result = result + normalized_coeff * Ylm
        
        return result
    
    @property
    def trainable_variables(self):
        """Return list of trainable variables for optimizer."""
        return [self.coefficients]


def find_next_run_number(output_dir) -> int:
    """
    Find the next run number for saving coefficients.
    
    Args:
        output_dir: Path to results directory
        
    Returns:
        int: Next run number
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    existing_runs = list(output_dir.glob("coefficients_run*.npz"))
    if existing_runs:
        run_numbers = []
        for f in existing_runs:
            try:
                num = int(f.stem.replace("coefficients_run", ""))
                run_numbers.append(num)
            except ValueError:
                pass
        return max(run_numbers) + 1 if run_numbers else 1
    return 1
