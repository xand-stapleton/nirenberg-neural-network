from abc import ABC, abstractmethod

import tensorflow as tf


class LossTerm(ABC):
    """
    Abstract base class for constituent loss terms.

    Each loss term operates on a batch dictionary, extracts relevant tensors,
    computes a loss value, and stores it back in the dictionary.

    Args:
        name: loss term name tag

    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, batch_dict: dict) -> dict:
        """
        Loss term logic to be implemented by subclasses.
        """
        pass


class ScalarLoss(LossTerm):
    """
    Scalar loss term for matching predicted scalar curvature to prescribed curvature.

    Args:
        name: loss term name tag
        conformal_factor_key: key that points to conformal factor `u`
        laplace_beltrami_key: key that points to laplacian of `u` w.r.t. round metric
        label_key: key that points to values of prescribed curvature
        normalization_key: key that points to normalization constant

    """

    def __init__(
        self,
        name: str,
        conformal_factor_key: str,
        laplace_beltrami_key: str,
        label_key: str,
        normalization_key: str,
    ):
        super().__init__(name)

        self.conformal_factor_key = conformal_factor_key
        self.laplace_beltrami_key = laplace_beltrami_key
        self.label_key = label_key
        self.normalization_key = normalization_key

    def __call__(self, batch_dict: dict) -> dict:
        # Extract input keys
        u = batch_dict[self.conformal_factor_key]
        delta_u = batch_dict[self.laplace_beltrami_key]
        label = batch_dict[self.label_key]
        normalizer = batch_dict[self.normalization_key]

        # Compute the predicted scalar curvature
        # For the round metric g0 (constant curvature 2), the scalar curvature of
        # the conformal metric g = e^(2u) * g0 is: R_g = e^(-2u) * (2 - Δ_g0 u)
        R_g = tf.exp(-2.0 * u) * (2.0 - 2.0 * delta_u)

        # MSE loss (normalizer set to 1.0 at dataset creation if too small)
        mse = tf.reduce_mean(tf.square(R_g - label) / normalizer)
        batch_dict[self.name] = mse

        return batch_dict


class CompoundLoss:
    """
    Combine multiple loss terms with specified multipliers.

    Args:
        name: compound loss name tag

    """

    def __init__(
        self,
        name: str,
        terms: list[LossTerm],
        multipliers: list[float],
        dtype: tf.DType,
    ):
        self.name = name
        self.terms = terms
        self.multipliers = multipliers
        self.dtype = dtype

    def __call__(self, batch_dict: dict) -> dict:
        total = tf.cast(0.0, self.dtype)
        for t, m in zip(self.terms, self.multipliers):
            batch_dict = t(batch_dict)
            total += tf.cast(m, self.dtype) * batch_dict[t.name]
        batch_dict[self.name] = total
        return batch_dict
