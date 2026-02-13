from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from geometry.ball import r_to_R, xyz_to_patch_xy, patch_xy_to_xyz


class Sampler(ABC):
    """
    Template class for various samplers.
    """

    def __init__(self, num_samples: int, dimension: int):
        self.num_samples = num_samples
        self.dimension = dimension

    @abstractmethod
    def __call__(self) -> tf.Tensor:
        """
        Sampling logic to be implemented by subclasses.

        Returns
        -------
        tf.Tensor
            generated samples, of shape (..., num_samples, dimension)
        """
        pass


class StereoSampler(Sampler):
    """
    Sample points uniformly on one or both spherical caps of S^2 and map them to 
    discs via standard stereographic projection from the north pole.

    This sampler uses standard stereographic projection where the radius in the disc
    is r = tan(θ/2) for a point at angle θ from the north pole. With radial_offset=0,
    the northern hemisphere (z >= 0) maps bijectively onto the unit disc.

    Args:
        num_patches: can be 1 (just northern patch) or 2 (both patches)
        num_samples: number of samples to generate per patch
        radial_offset: angular offset in radians to extend sampling beyond π/2 (hemisphere).
            With radial_offset > 0, samples points with θ ∈ [0, π/2 + radial_offset],
            which extends into the southern hemisphere (z < 0) and produces disc radii > 1
        dtype: data type for tensor computations

    """

    def __init__(
        self,
        num_patches: int = 2,
        num_samples: int = 1000,
        radial_offset: float = 0.0,
        dtype: tf.DType = tf.float64,
    ):
        super().__init__(num_samples, dimension=2)
        
        if num_patches not in (1, 2):
            raise ValueError("num_patches must be 1 or 2")
        
        if radial_offset < 0.0:
            raise ValueError("radial_offset must be non-negative")
        
        self.num_patches = num_patches
        self.radial_offset = radial_offset
        self.dtype = dtype

    def _sample_cap(self, z_min: float, z_max: float, patch_idx: int) -> np.ndarray:
        """
        Sample uniformly on the northern hemisphere and project to disc via standard
        stereographic projection from the north pole.

        Note: The z_min and z_max parameters are currently unused. The implementation
        samples the full northern hemisphere (xi_3 in [0,1]) and projects using the
        formula X = xi / (1 + xi_3) where xi is the point on the sphere.

        Args:
            z_min: minimum z-coordinate (unused in current implementation)
            z_max: maximum z-coordinate (unused in current implementation)
            patch_idx: 0 (north patch) or 1 (south patch) - determines orientation

        Returns:
            np.ndarray: (num_samples, 2)
                2D disc coordinates via stereographic projection from north pole

        """
        # Step 1: sample hemisphere uniformly
        u = np.random.uniform(0.0, 1.0, size=self.num_samples)  # cos(theta) in [0,1]
        theta = np.arccos(u)
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=self.num_samples)

        # Step 2: spherical to Cartesian
        xi_1 = np.sin(theta) * np.cos(phi)
        xi_2 = np.sin(theta) * np.sin(phi)
        xi_3 = np.cos(theta)  # in [0,1]

        # Step 3: stereographic projection from NORTH pole -> unit disc
        denom = 1.0 + xi_3  # never zero on hemisphere (xi_3 >= 0)
        X_1 = xi_1 / denom
        X_2 = xi_2 / denom

        return np.stack([X_1, X_2], axis=1)

    def __call__(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generate samples on one or both patches via stereographic projection.
        
        The method samples disc coordinates directly via stereographic projection,
        then converts them back to 3D sphere coordinates using patch_xy_to_xyz.
        This ensures consistency between the 2D and 3D representations.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: A tuple containing:
                - Disc coordinates: (num_samples, 2) if num_patches=1,
                                   (num_samples, num_patches, 2) if num_patches=2
                - Sphere coordinates: (num_samples, 3) if num_patches=1,
                                     (num_samples, num_patches, 3) if num_patches=2

        """
        # Determine z-range based on radial_offset
        # theta_max = π/2 + radial_offset
        theta_max = np.pi / 2.0 + self.radial_offset
        z_min = np.cos(theta_max)  # can be negative if radial_offset > 0

        patches, caps = [], []
        
        # NOTE: Using standard stereographic projection (matching old branch):
        # - patch_idx=0: projects FROM south pole → covers NORTHERN hemisphere (z > 0)
        # - patch_idx=1: projects FROM north pole → covers SOUTHERN hemisphere (z < 0)
        
        # Sample northern cap (z >= 0) and use patch_idx=0
        uv_north = self._sample_cap(z_min, 1.0, patch_idx=0)
        uv_north_tensor = tf.convert_to_tensor(uv_north, dtype=self.dtype)
        xyz_north = patch_xy_to_xyz(uv_north_tensor, patch_idx=0)
        patches.append(uv_north_tensor)
        caps.append(xyz_north)

        if self.num_patches == 2:
            # Sample southern cap (z <= 0) and use patch_idx=1
            uv_south = self._sample_cap(-1.0, -z_min, patch_idx=1)
            uv_south_tensor = tf.convert_to_tensor(uv_south, dtype=self.dtype)
            xyz_south = patch_xy_to_xyz(uv_south_tensor, patch_idx=1)
            patches.append(uv_south_tensor)
            caps.append(xyz_south)

        if self.num_patches == 1:
            return patches[0], caps[0]
        else:
            # Stack along patch dimension: (num_samples, num_patches, dims)
            return tf.stack(patches, axis=1), tf.stack(caps, axis=1)


# COMMENTED OUT: Old version with modified stereographic projection
# class StereoSamplerModified(Sampler):
#     """
#     Sample one or both of the 2D discs constituting the standard two-chart atlas on the
#     sphere using modified stereographic projection.
#
#     Args:
#         num_patches: can be 1 (just northern patch) or 2 (both patches)
#         num_samples: number of samples per patch
#         radial_offset: offset to be added to the midpoint radius
#         dtype: data type
#
#     """
#
#     MIDPOINT_RADIUS = np.sqrt(2) - 1
#
#     def __init__(
#         self,
#         num_patches: int,
#         num_samples: int,
#         radial_offset: float = 0.1,
#         dtype: tf.DType = tf.float64,
#     ):
#         super().__init__(num_samples, dimension=2)
#
#         if num_patches not in (1, 2):
#             raise ValueError("num_patches must be 1 or 2")
#
#         if not (radial_offset >= 0 and radial_offset < 1.0 - self.MIDPOINT_RADIUS):
#             raise ValueError(
#                 f"radial_offset outside of valid bounds: [0, {1.0 - self.MIDPOINT_RADIUS})"
#             )
#
#         self.num_patches = num_patches
#         self.radial_offset = radial_offset
#         self.dtype = dtype
#
#     def sample_cap(self, z_cutoff: float, patch_idx: int) -> tf.Tensor:
#         """
#         Uniformly sample the sphere (in xyz coordinates) starting from a pole up to a
#         z-coordinate cutoff between -1 and 1.
#
#         Args:
#             z_cutoff: if sampling northern patch, interpreted as a lower bound; if sampling
#                 southern patch, inerpreted as an upper bound
#             patch_idx: 0 (north) or 1 (south)
#
#         Returns:
#             tf.Tensor: (num_samples, 3)
#                 sampled cap
#
#         """
#         # Uniformly sample z-coordinates
#         if patch_idx == 0:
#             z = np.random.uniform(z_cutoff, 1.0, self.num_samples)
#         elif patch_idx == 1:
#             z = np.random.uniform(-1.0, z_cutoff, self.num_samples)
#         else:
#             raise ValueError
#
#         phi = np.random.uniform(0.0, 2.0 * np.pi, self.num_samples)
#         r_xy = np.sqrt(1.0 - z**2)
#         x, y = r_xy * np.cos(phi), r_xy * np.sin(phi)
#         xyz = np.stack([x, y, z], axis=-1)
#         return tf.convert_to_tensor(xyz, dtype=self.dtype)
#
#     def __call__(self) -> tuple[tf.Tensor, tf.Tensor]:
#         """
#         Sampling logic.
#
#         Returns:
#             tf.Tensor: (num_samples, num_patches, 2)
#                 sampled patch(es)
#             tf.Tensor: (num_samples, num_patches, 3)
#                 sampled cap(s)
#
#         """
#         # Maximum allowable disc radius. We don't want to sample near the disc boundary
#         r_max = self.MIDPOINT_RADIUS + self.radial_offset
#         # The maximum allowable disc radius corresponds to a minimum allowable z-coordinate
#         # for the northern patch. We compute this as z_min = (1 - R_max^2) / (1 + R_max^2)
#         R_max = r_to_R(r_max)
#         z_min = (1.0 - R_max**2) / (1.0 + R_max**2)
#
#         patches, caps = [], []
#         # Sample northern cap in R^3 coordinates
#         xyz = self.sample_cap(z_min, patch_idx=0)
#         caps.append(xyz)
#         # Convert to patch coordinates (sphere -> stereo -> disc)
#         uv = xyz_to_patch_xy(xyz, patch_idx=0)
#         patches.append(uv)
#
#         if self.num_patches == 2:
#             # Sample southern cap in R^3 coordinates
#             xyz = self.sample_cap(-z_min, patch_idx=1)
#             caps.append(xyz)
#             # Convert to patch coordinates (sphere -> stereo -> disc)
#             uv = xyz_to_patch_xy(xyz, patch_idx=1)
#             patches.append(uv)
#
#         return tf.stack(patches, axis=1), tf.stack(caps, axis=1)
