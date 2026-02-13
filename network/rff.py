from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="RandomFourierFeature")
class RandomFourierFeature(Layer):
    """
    Random Fourier Feature encoding layer.

    Implements the formulation introduced by Rahimi and Recht in NIPS 2007. The encoding
    is given by φ(x) = √(2/m) * [cos(Fx + ω), sin(Fx + ω)], where F is a random frequency
    matrix, ω is a random phase vector, and m is a hyperparameter specifying the size of
    the encoding.

    Args:
        num_features: output encoding will have length 2 * num_features
        sigma: spread of gaussian from which we sample entries of F

    """

    def __init__(
        self,
        num_features: int,
        sigma: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_features = num_features
        self.sigma = sigma

    def build(self, input_shape: tuple[int, ...] | tf.TensorShape) -> None:
        """
        Construct the random frequency matrix F and phase vector ω.

        Args:
            input_shape: shape of input tensor (B, input_dim)

        """
        input_dim = input_shape[-1]

        # Random frequency matrix F
        self.F: tf.Variable = self.add_weight(
            name="frequency_matrix",
            shape=(input_dim, self.num_features),
            initializer=RandomNormal(stddev=self.sigma),
            trainable=False,
            dtype=self.dtype,
        )

        # Random phase vector ω
        self.ω: tf.Variable = self.add_weight(
            name="phase_vector",
            shape=(self.num_features,),
            initializer=RandomUniform(minval=0.0, maxval=2.0 * np.pi),
            trainable=False,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """
        Apply random Fourier feature encoding.

        Args:
            inputs: (B, input_dim)
                input coordinates
            training: training mode

        Returns:
            tf.Tensor: (B, 2 * num_features)
                encoded features

        """
        features = inputs @ self.F + self.ω[None, :]
        cos_features = tf.cos(features)
        sin_features = tf.sin(features)
        rf_features = tf.concat([cos_features, sin_features], axis=-1)
        return tf.sqrt(2.0 / tf.cast(self.num_features, self.dtype)) * rf_features

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            dict: configuration dictionary

        """
        config = super().get_config()
        config.update({"num_features": self.num_features, "sigma": self.sigma})
        return config
