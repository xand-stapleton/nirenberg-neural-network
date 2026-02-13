import math
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from geometry.integration import integrate_gauss_legendre, integrate_monte_carlo
from schemas.enums import Solvable


def build_prescriber(kind: str, **kwargs) -> "Prescriber":
    """
    Factory function to create a prescriber instance by name.

    Args:
        kind: name of the prescriber class (e.g., "round", "cos", "gaussian_bump")
        **kwargs: keyword arguments passed to the prescriber constructor

    Returns:
        Prescriber: instance of the requested prescriber class

    Raises:
        ValueError: if kind is not found in the registry
    """
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown prescriber kind: {kind!r}. Available: {list(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


class Prescriber(ABC):
    """
    A prescribed scalar curvature function.

    Args:
        solvable: whether it is known (YES, NO, UNKNOWN) to admit a solution to the
            Kazdan-Warner problem

    """

    def __init__(self, solvable: Solvable):
        self.solvable = solvable

    @abstractmethod
    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        """
        Definition of function in terms of angular coordinates. To be implemented by
        subclass.

        Args:
            theta: (..., 1)
            phi: (..., 1)

        Returns:
            tf.Tensor: (..., 1)
                prescribed scalar curvature

        """
        pass

    def __call__(self, xyz: tf.Tensor) -> tf.Tensor:
        """
        Evaluate function on Cartesian coordinates.

        Args:
            xyz: (..., 3)
                Cartesian coordinates at which to evaluate

        Returns:
            tf.Tensor: (..., 1)
                prescribed scalar curvature

        """
        theta, phi = self.cart_to_angular(xyz)
        return self.forward(theta, phi)

    def cart_to_angular(self, xyz: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Convert Cartesian coordinates to angular coordinates on the unit sphere.

        Args:
            xyz: (..., 3)
                Cartesian coordinates

        Returns:
            tf.Tensor: (..., 1)
                theta
            tf.Tensor: (..., 1)
                phi

        """
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        theta = tf.acos(tf.clip_by_value(z, -1.0, 1.0))
        phi = tf.math.atan2(y, x)
        two_pi = tf.cast(2.0 * np.pi, xyz.dtype)
        phi = tf.where(phi < 0.0, phi + two_pi, phi)
        return theta[..., None], phi[..., None]

    def integrate(
        self,
        integrand: str = "square",
        method: str = "monte_carlo",
        **kwargs,
    ) -> float:
        """
        Compute the integral of the prescriber function over the sphere.

        Args:
            integrand: type of integrand: 'self', 'abs' or 'square'
            method: method of numerical integration: 'monte_carlo' or 'gauss_legendre'
            kwargs: additional arguments passed to the integration routine

        Returns:
            float: integral over the sphere

        """
        # Set integrand
        if integrand == "self":
            f = self.forward
        elif integrand == "abs":
            f = lambda theta, phi: tf.abs(self.forward(theta, phi))  # noqa: E731
        elif integrand == "square":
            f = lambda theta, phi: tf.square(self.forward(theta, phi))  # noqa: E731
        else:
            raise ValueError(
                f"Unknown integrand type: {integrand!r}.Available: 'self', 'abs', 'square'."
            )

        # Dispatch integration method
        if method == "monte_carlo":
            return integrate_monte_carlo(f, **kwargs)
        elif method == "gauss_legendre":
            return integrate_gauss_legendre(f, **kwargs)
        else:
            raise ValueError(
                f"Unknown integration method: {method!r}."
                f"Available: 'monte_carlo', 'gauss_legendre'."
            )


class Round(Prescriber):
    """
    Constant scalar curvature R = 2 (the round metric itself).

    This is the trivial case where the conformal factor u = 0 solves the problem.
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(theta) * tf.constant(2.0, dtype=tf.float64)

class Zero(Prescriber):
    """
    Constant scalar curvature R = 0.

    This is the trivial case which cannot be solved since there're no Ricci-flat metrics on S^2.
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(theta)

class Cos2ThetaPlusOffset(Prescriber):
    """
    Prescribed curvature R(θ) = cos²(θ) + offset.

    Args:
        offset: constant offset added to cos²(θ)
    """

    def __init__(self, offset: float = 0.2):
        super().__init__(Solvable.YES)
        self.offset = offset

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.math.cos(theta) ** 2 + tf.cast(self.offset, theta.dtype)


class Sinusoidal(Prescriber):
    """
    Sinusoidal variation: R(θ) = 1 + 0.5 * sin(3θ).

    This is a smooth, oscillatory function that admits a solution.
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return 1.0 + 0.5 * tf.sin(3.0 * theta)


class ThreeCos2(Prescriber):
    """
    Second Legendre polynomial: R(θ) = 3cos²(θ) - 1.

    This is proportional to the l=2, m=0 spherical harmonic.
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return 3.0 * tf.math.cos(theta) ** 2 - 1.0


class FiveCos3(Prescriber):
    """
    Third Legendre polynomial: R(θ) = 5cos³(θ) - 3cos(θ).

    This is proportional to the l=3, m=0 spherical harmonic.
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        c = tf.math.cos(theta)
        return 5.0 * c**3 - 3.0 * c


class FiveCos3Plus1(Prescriber):
    """
    Third Legendre polynomial with constant offset: R(θ) = 5cos³(θ) - 3cos(θ) + 1.

    The constant offset ensures this function admits a solution.
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        c = tf.math.cos(theta)
        return 5.0 * c**3 - 3.0 * c + 1.0


class TwoPlusX(Prescriber):
    """
    Prescribed curvature R(x) = 2 + x

    NOTE:

    2 + x may be described as sin(theta) cos(phi)
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.cast(2.0, theta.dtype) + tf.math.sin(theta) * tf.math.cos(phi)


class TwoPlusY(Prescriber):
    """
    Prescribed curvature R(x) = 2 + y

    NOTE:

    2 + y may be described as sin(theta) sin(phi)
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.cast(2.0, theta.dtype) + tf.math.sin(theta) * tf.math.sin(phi)


class ZPlusXSquaredOverFour(Prescriber):
    """
    Prescribed curvature R(x) = z+1/4 x^2

    NOTE:

    z + 1/4 x^2 may be written as cos(theta) + sin^2(theta) cos^2(phi)
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.square(tf.math.sin(theta) * tf.math.sin(phi)) + tf.math.cos(theta)


class ZPlusXY(Prescriber):
    """
    Prescribed curvature R(x) = z+xy

    NOTE:

    z + xy may be written as cos(theta) + sin^2(theta)(cos(phi) sin(phi)
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.math.cos(theta) + (
            tf.square(tf.math.sin(theta)) * tf.math.sin(phi) * tf.math.cos(phi)
        )


class Cosh(Prescriber):
    """
    Azimuthally symmetric profile:
    R(θ) = 2 cosh(cos θ)
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return 2.0 * tf.math.cosh(tf.math.cos(theta))


class Egg(Prescriber):
    """
    Egg-shaped azimuthally symmetric profile:
    R(θ) = (1.5 + cos θ)^2 (1.2 − cos θ)
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        cos_t = tf.math.cos(theta)
        return tf.math.square(1.5 + cos_t) * (1.2 - cos_t)


class TanhWave(Prescriber):
    """
    Smooth wave-like profile:
    R(θ) = tanh(2 cos² θ) + cos² θ + 0.5
    """

    def __init__(self):
        super().__init__(Solvable.YES)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        cos2 = tf.square(tf.math.cos(theta))
        return tf.math.tanh(2.0 * cos2) + cos2 + tf.cast(0.5, theta.dtype)


class NodalCrossing(Prescriber):
    """
    Azimuthally symmetric profile:
    R(θ) = cos^2(θ) - 0.9
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.square(tf.math.cos(theta)) - 0.9


class NegativeDip(Prescriber):
    """
    Azimuthally symmetric profile:
    R(θ) = -1 / (1 + cos^2(θ))
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return -1.0 / (1.0 + tf.square(tf.math.cos(theta)))


class MonotonicExp(Prescriber):
    """
    Azimuthally symmetric profile:
    R(θ) = exp(2 cos(θ))
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(2.0 * tf.math.cos(theta))


class CosThetaPlusSin2ThetaCosPhiOverFour(Prescriber):
    """
    Prescribed curvature R(θ, φ) = cos(θ) + (1/4)sin²(θ)cos(φ).
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_theta = tf.math.cos(theta)
        cos_phi = tf.math.cos(phi)
        return cos_theta + 0.25 * tf.square(sin_theta) * cos_phi


class CosThetaPlusSin2ThetaSinPhiOverFour(Prescriber):
    """
    Prescribed curvature R(θ, φ) = cos(θ) + (1/4)sin²(θ)sin(φ).
    """

    def __init__(self):
        super().__init__(Solvable.NO)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_theta = tf.math.cos(theta)
        sin_phi = tf.math.sin(phi)
        return cos_theta + 0.25 * tf.square(sin_theta) * sin_phi


class X2MinusY2PlusZ(Prescriber):
    """
    Prescribed curvature R(x, y, z) = x² - y² + z.

    In spherical coordinates: sin²(θ)cos²(φ) - sin²(θ)sin²(φ) + cos(θ).
    """

    def __init__(self):
        super().__init__(Solvable.UNKNOWN)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_theta = tf.math.cos(theta)
        sin_phi = tf.math.sin(phi)
        cos_phi = tf.math.cos(phi)
        return tf.square(sin_theta) * (tf.square(cos_phi) - tf.square(sin_phi)) + cos_theta


class X2Plus2Y2Plus3Z2PlusXPlusY(Prescriber):
    """
    Prescribed curvature R(x, y, z) = x² + 2y² + 3z² + x + y.

    In spherical coordinates:
    sin²(θ)cos²(φ) + 2sin²(θ)sin²(φ) + 3cos²(θ) + sin(θ)cos(φ) + sin(θ)sin(φ).
    """

    def __init__(self):
        super().__init__(Solvable.UNKNOWN)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_theta = tf.math.cos(theta)
        sin_phi = tf.math.sin(phi)
        cos_phi = tf.math.cos(phi)
        return (
            tf.square(sin_theta) * tf.square(cos_phi)
            + 2.0 * tf.square(sin_theta) * tf.square(sin_phi)
            + 3.0 * tf.square(cos_theta)
            + sin_theta * cos_phi
            + sin_theta * sin_phi
        )


class X2PlusYZPlusX(Prescriber):
    """
    Prescribed curvature R(x, y, z) = x² + yz + x.

    In spherical coordinates: sin²(θ)cos²(φ) + sin(θ)sin(φ)cos(θ) + sin(θ)cos(φ).
    """

    def __init__(self):
        super().__init__(Solvable.UNKNOWN)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_theta = tf.math.cos(theta)
        sin_phi = tf.math.sin(phi)
        cos_phi = tf.math.cos(phi)
        return (
            tf.square(sin_theta) * tf.square(cos_phi)
            + sin_theta * sin_phi * cos_theta
            + sin_theta * cos_phi
        )


class X2PlusYZPlusXPlus1(Prescriber):
    """
    Prescribed curvature R(x, y, z) = x² + yz + x + 1.

    In spherical coordinates: sin²(θ)cos²(φ) + sin(θ)sin(φ)cos(θ) + sin(θ)cos(φ) + 1.
    """

    def __init__(self):
        super().__init__(Solvable.UNKNOWN)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_theta = tf.math.cos(theta)
        sin_phi = tf.math.sin(phi)
        cos_phi = tf.math.cos(phi)
        return (
            tf.square(sin_theta) * tf.square(cos_phi)
            + sin_theta * sin_phi * cos_theta
            + sin_theta * cos_phi
            + 1.0
        )


class XYPlusXPlusY(Prescriber):
    """
    Prescribed curvature R(x, y, z) = xy + x + y.

    In spherical coordinates: sin²(θ)cos(φ)sin(φ) + sin(θ)cos(φ) + sin(θ)sin(φ).
    """

    def __init__(self):
        super().__init__(Solvable.UNKNOWN)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_phi = tf.math.cos(phi)
        sin_phi = tf.math.sin(phi)
        return (
            tf.square(sin_theta) * cos_phi * sin_phi
            + sin_theta * cos_phi
            + sin_theta * sin_phi
        )


class Y2PlusXZPlusZMinus1(Prescriber):
    """
    Prescribed curvature R(x, y, z) = y² + xz + z - 1.

    In spherical coordinates: sin²(θ)sin²(φ) + sin(θ)cos(φ)cos(θ) + cos(θ) - 1.
    """

    def __init__(self):
        super().__init__(Solvable.UNKNOWN)

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        sin_theta = tf.math.sin(theta)
        cos_theta = tf.math.cos(theta)
        sin_phi = tf.math.sin(phi)
        cos_phi = tf.math.cos(phi)
        return (
            tf.square(sin_theta) * tf.square(sin_phi)
            + sin_theta * cos_phi * cos_theta
            + cos_theta
            - 1.0
        )


class SphericalHarmonics(Prescriber):
    """
    Prescribed curvature as a single real spherical harmonic Y_l^m(θ, φ).

    Args:
        l: frequency (degree) l >= 0
        m: order with |m| <= l

    """

    def __init__(self, l: int, m: int = 0):
        # Determine solvability based on l and m values
        if l == 1 and m == 0:
            # Y_1^0 is proportional to cos(θ), which is known to have no solution
            solvable = Solvable.NO
        elif m == 0:
            # Other Y_l^0 cases with l != 1 are known to be solvable
            solvable = Solvable.YES
        elif l%2 == 0:
            # Other Y_l^m cases where l is even are known to be solvable
            solvable = Solvable.YES
        else:
            # Other cases are unknown
            solvable = Solvable.UNKNOWN
        
        super().__init__(solvable)
        self.l = l
        self.m = m

        if abs(m) > l:
            raise ValueError(f"|m| must be <= l, got m={m}, l={l}")

    def _associated_legendre_polynomial(
        self,
        l: int,  # noqa: E741
        m: int,
        x: tf.Tensor,
    ) -> tf.Tensor:
        """
        Evaluate the associated Legendre polynomial P_l^|m| at points x using the stable
        three-term recurrence.

        Args:
            l: non-negative degree
            m: order with |m| <= l
            x: (...,)
                tensor with entries inside [-1, 1].

        Returns:
            tf.Tensor: (...,)
                values of P_l^|m| at x

        """
        # Use absolute value of order. Negative orders reuse same ALP as positive orders
        # in the case of real spherical harmonics
        m_abs = abs(m)
        # (1 - x^2) is a shared subexpression across all diagonal ALPs of the form P_m^m
        one_minus_x2 = 1.0 - tf.square(x)

        if m_abs == 0:
            # Base case P_0^0 = 1 for every x
            Pmm = tf.ones_like(x)
        else:
            # Compute the double-factorial coefficient (2m-1)!! that appears in P_m^m
            df = math.factorial(2 * m_abs) // (2**m_abs * math.factorial(m_abs))
            df = tf.cast(df, x.dtype)
            # Compute the Condon-Shortley phase
            cs = (-1) ** m_abs
            cs = tf.cast(cs, x.dtype)
            # Compute P_m^m
            Pmm = cs * df * tf.pow(one_minus_x2, tf.cast(0.5 * m_abs, x.dtype))

        if l == m_abs:
            # No recurrence needed when l == |m|
            return Pmm

        # Seed values for the recurrence:
        # P_{l-2}^m
        Pnm2 = Pmm
        # P_{l-1}^m: obtained from relation [P_{m+1}^m](x) = x * (2m + 1) * [P_m^m](x)
        Pnm1 = x * tf.cast(2.0 * m_abs + 1.0, x.dtype) * Pmm

        if l == m_abs + 1:
            # No recurrence needed when l == |m| + 1
            return Pnm1

        # Apply the recurrence relation:
        # (n - m) * [P_n^m](x) = x * (2n - 1) * [P_{n-1}^m](x) - (n + m - 1) * [P_{n-2}^m](x)
        for n in range(m_abs + 2, l + 1):
            term1 = tf.cast(2.0 * n - 1.0, x.dtype) * x * Pnm1
            term2 = tf.cast(n + m_abs - 1.0, x.dtype) * Pnm2
            Pn = (term1 - term2) / tf.cast(n - m_abs, x.dtype)
            Pnm2, Pnm1 = Pnm1, Pn

        return Pnm1

    def _real_spherical_harmonic(
        self,
        l: int,  # noqa: E741
        m: int,
        theta: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the real-valued spherical harmonic [Y_l^m](θ, φ).

        Args:
            l: non-negative degree
            m: order with |m| <= l
            theta: (..., 1)
                colatitudes in radians
            phi: (..., 1)
                longitudes in radians

        Returns:
            tf.Tensor: (..., 1)
                values of real spherical harmonic at (theta, phi)

        """
        if abs(m) > l:
            raise ValueError(f"|m| must be <= l, got m={m}, l={l}")

        # Associated Legendre polynomial supplies the θ dependence
        x = tf.cos(theta)
        Plm = self._associated_legendre_polynomial(l, m, x)

        # Normalization factor from spherical-harmonic convention
        prefactor = (2.0 * l + 1.0) / (4.0 * np.pi)
        a = tf.math.lgamma(tf.cast(l - abs(m) + 1, theta.dtype))
        b = tf.math.lgamma(tf.cast(l + abs(m) + 1, theta.dtype))
        log_ratio = a - b
        Nlm = tf.sqrt(tf.cast(prefactor, theta.dtype) * tf.exp(log_ratio))

        # Branch on m to select cosine (m>0), sine (m<0), or constant (m=0) form.
        if m == 0:
            return Nlm * Plm
        elif m > 0:
            cos_mphi = tf.cos(tf.cast(m, phi.dtype) * phi)
            return tf.cast(np.sqrt(2.0), theta.dtype) * Nlm * Plm * cos_mphi
        else:
            sin_mphi = tf.sin(tf.cast(abs(m), phi.dtype) * phi)
            return tf.cast(np.sqrt(2.0), theta.dtype) * Nlm * Plm * sin_mphi

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        # Return the spherical harmonic Y_l^m as the prescribed curvature
        return self._real_spherical_harmonic(self.l, self.m, theta, phi)


class SpectralPair(Prescriber):
    """
    Prescribed scalar curvature that is built from a finite real spherical-harmonic
    expansion, as described in Proposition 2.2.

    Each harmonic term uses frequency l, order m, and amplitude c. The resulting
    curvature serves both as a known Kazdan–Warner solution and as a convenient
    benchmark because its conformal factor is available in closed form.

    Args:
        ls: sequence of frequencies l >= 1
        cs: matching amplitudes (real)
        ms: matching orders |m| <= l

    """

    def __init__(self, ls: list[int], cs: list[float], ms: list[int] | None = None):
        super().__init__(Solvable.YES)
        self.ls = ls
        self.cs = cs
        self.ms = ms if ms is not None else [0] * len(ls)

        if len(self.ls) != len(self.cs) or len(self.ls) != len(self.ms):
            raise ValueError(
                f"Lengths must match: ls={len(self.ls)}, cs={len(self.cs)}, ms={len(self.ms)}"
            )

    def _associated_legendre_polynomial(
        self,
        l: int,  # noqa: E741
        m: int,
        x: tf.Tensor,
    ) -> tf.Tensor:
        """
        Evaluate the associated Legendre polynomial P_l^|m| at points x using the stable
        three-term recurrence.

        Args:
            l: non-negative degree
            m: order with |m| <= l
            x: (...,)
                tensor with entries inside [-1, 1].

        Returns:
            tf.Tensor: (...,)
                values of P_l^|m| at x

        """
        # Use absolute value of order. Negative orders reuse same ALP as positive orders
        # in the case of real spherical harmonics
        m_abs = abs(m)
        # (1 - x^2) is a shared subexpression across all diagonal ALPs of the form P_m^m
        one_minus_x2 = 1.0 - tf.square(x)

        if m_abs == 0:
            # Base case P_0^0 = 1 for every x
            Pmm = tf.ones_like(x)
        else:
            # Compute the double-factorial coefficient (2m-1)!! that appears in P_m^m
            df = math.factorial(2 * m_abs) // (2**m_abs * math.factorial(m_abs))
            df = tf.cast(df, x.dtype)
            # Compute the Condon-Shortley phase
            cs = (-1) ** m_abs
            cs = tf.cast(cs, x.dtype)
            # Compute P_m^m
            Pmm = cs * df * tf.pow(one_minus_x2, tf.cast(0.5 * m_abs, x.dtype))

        if l == m_abs:
            # No recurrence needed when l == |m|
            return Pmm

        # Seed values for the recurrence:
        # P_{l-2}^m
        Pnm2 = Pmm
        # P_{l-1}^m: obtained from relation [P_{m+1}^m](x) = x * (2m + 1) * [P_m^m](x)
        Pnm1 = x * tf.cast(2.0 * m_abs + 1.0, x.dtype) * Pmm

        if l == m_abs + 1:
            # No recurrence needed when l == |m| + 1
            return Pnm1

        # Apply the recurrence relation:
        # (n - m) * [P_n^m](x) = x * (2n - 1) * [P_{n-1}^m](x) - (n + m - 1) * [P_{n-2}^m](x)
        for n in range(m_abs + 2, l + 1):
            term1 = tf.cast(2.0 * n - 1.0, x.dtype) * x * Pnm1
            term2 = tf.cast(n + m_abs - 1.0, x.dtype) * Pnm2
            Pn = (term1 - term2) / tf.cast(n - m_abs, x.dtype)
            Pnm2, Pnm1 = Pnm1, Pn

        return Pnm1

    def _real_spherical_harmonic(
        self,
        l: int,  # noqa: E741
        m: int,
        theta: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the real-valued spherical harmonic [Y_l^m](θ, φ).

        Args:
            l: non-negative degree
            m: order with |m| <= l
            theta: (..., 1)
                colatitudes in radians
            phi: (..., 1)
                longitudes in radians

        Returns:
            tf.Tensor: (..., 1)
                values of real spherical harmonic at (theta, phi)

        """
        if abs(m) > l:
            raise ValueError(f"|m| must be <= l, got m={m}, l={l}")

        # Associated Legendre polynomial supplies the θ dependence
        x = tf.cos(theta)
        Plm = self._associated_legendre_polynomial(l, m, x)

        # Normalization factor from spherical-harmonic convention
        prefactor = (2.0 * l + 1.0) / (4.0 * np.pi)
        a = tf.math.lgamma(tf.cast(l - abs(m) + 1, theta.dtype))
        b = tf.math.lgamma(tf.cast(l + abs(m) + 1, theta.dtype))
        log_ratio = a - b
        Nlm = tf.sqrt(tf.cast(prefactor, theta.dtype) * tf.exp(log_ratio))

        # Branch on m to select cosine (m>0), sine (m<0), or constant (m=0) form.
        if m == 0:
            return Nlm * Plm
        elif m > 0:
            cos_mphi = tf.cos(tf.cast(m, phi.dtype) * phi)
            return tf.cast(np.sqrt(2.0), theta.dtype) * Nlm * Plm * cos_mphi
        else:
            sin_mphi = tf.sin(tf.cast(abs(m), phi.dtype) * phi)
            return tf.cast(np.sqrt(2.0), theta.dtype) * Nlm * Plm * sin_mphi

    def _conformal_factor(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        """
        Compute the conformal factor u in closed form, as per Theorem 2.1.

        Args:
            theta: (..., 1)
                colatitude angle in [0, π]
            phi: (..., 1)
                azimuthal angle in [0, 2π)

        Returns:
            tf.Tensor: (..., 1)
                conformal factor u

        """
        u = tf.zeros_like(theta)

        # Accumulate contributions from each spherical-harmonic term (l, m, c)
        for l, m, c in zip(self.ls, self.ms, self.cs):  # noqa: E741
            # Evaluate the real spherical harmonic [Y_l^m](θ, φ)
            Ylm = self._real_spherical_harmonic(l, m, theta, phi)
            # Cast coefficient and degree to match the computation dtype.
            c_tf = tf.cast(c, theta.dtype)
            l_tf = tf.cast(l, theta.dtype)
            # Add the weighted harmonic: u ← u + (c_l / (l(l+1))) * Y_l^m
            u = u + (c_tf / (l_tf * (l_tf + 1.0))) * Ylm

        return u

    def forward(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        # Compute conformal factor u(θ, φ) = Σ c_l / (l(l+1)) * Y_l^m
        u = self._conformal_factor(theta, phi)
        # Initialize (1 + Σ c_l Y_l^m)
        total_base = tf.ones_like(theta)

        # Accumulate curvature amplitudes Σ c_l [Y_l^m](θ, φ)
        for l, m, c in zip(self.ls, self.ms, self.cs):  # noqa: E741
            # Evaluate the real spherical harmonic [Y_l^m](θ, φ)
            Ylm = self._real_spherical_harmonic(l, m, theta, phi)
            # Cast coefficient to match the computation dtype
            c = tf.cast(c, theta.dtype)
            # Update (1 + Σ c_l Y_l^m)
            total_base = total_base + c * Ylm

        # Scalar curvature:
        # R(θ, φ) = 2 * (1 + Σ c_l Y_l^m(θ, φ)) * exp(-2 u(θ, φ)).
        return 2.0 * total_base * tf.exp(-2.0 * u)


if __name__ == "__main__":
    prescriber = SpectralPair(ls=[1, 1, 1], cs=[1, 1, 1], ms=[-1, 0, 1])
    print(prescriber.integrate(integrand="abs", method="monte_carlo"))
    print(prescriber.integrate(integrand="abs", method="gauss_legendre"))
    print(prescriber.solvable)


_REGISTRY = {
    "3cos2": ThreeCos2,
    "5cos3": FiveCos3,
    "5cos3_1": FiveCos3Plus1,
    "cos2_theta_plus_offset": Cos2ThetaPlusOffset,
    "cos_theta_plus_sin2_theta_cos_phi_over_four": CosThetaPlusSin2ThetaCosPhiOverFour,
    "cos_theta_plus_sin2_theta_sin_phi_over_four": CosThetaPlusSin2ThetaSinPhiOverFour,
    "cosh_profile": Cosh,
    "egg": Egg,
    "monotonic_exp": MonotonicExp,
    "negative_dip": NegativeDip,
    "nodal_crossing": NodalCrossing,
    "round": Round,
    "sh": SphericalHarmonics,
    "sinusoidal": Sinusoidal,
    "tanh_wave": TanhWave,
    "prop": SpectralPair,
    "two_plus_x": TwoPlusX,
    "two_plus_y": TwoPlusY,
    "x2_minus_y2_plus_z": X2MinusY2PlusZ,
    "x2_plus_2y2_plus_3z2_plus_x_plus_y": X2Plus2Y2Plus3Z2PlusXPlusY,
    "x2_plus_yz_plus_x": X2PlusYZPlusX,
    "x2_plus_yz_plus_x_plus_1": X2PlusYZPlusXPlus1,
    "xy_plus_x_plus_y": XYPlusXPlusY,
    "y2_plus_xz_plus_z_minus_1": Y2PlusXZPlusZMinus1,
    "z_plus_x_squared_over_four": ZPlusXSquaredOverFour,
    "z_plus_xy": ZPlusXY,
    "zero": Zero,
}
