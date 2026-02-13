from typing import Any

import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="ResidualBlock")
class ResidualBlock(Layer):
    """
    Dense block with optional residual connection.

    If use_residual is:
        - True  : always use skip connection (requires matching dimensions)
        - False : behave as a simple MLP block
        - None  : automatically use residual only when shapes match
    """

    def __init__(
        self,
        num_hidden: int,
        activation: str = "silu",
        initializer: str = "he_uniform",
        use_bias: bool = True,
        use_residual: bool | None = None,
        dtype: tf.DType = tf.float64,
        **kwargs,
    ) -> None:
        super().__init__(dtype=dtype, **kwargs)
        self.num_hidden = num_hidden
        self.activation = activation
        self.initializer = initializer
        self.use_bias = use_bias
        self.use_residual = use_residual

    def build(self, input_shape: tf.TensorShape) -> None:

        # First dense layer (with activation)
        self.dense_1 = Dense(
            units=self.num_hidden,
            activation=self.activation,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_initializer=self.initializer,
            # bias_initializer=Constant(0.0),
            name=f"{self.name}_dense_1" if self.name else None,
        )

        # First dense layer (without activation)
        # Only used if using a residual layer
        self.dense_2 = Dense(
            units=self.num_hidden,
            activation=None,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_initializer=self.initializer,
            bias_initializer=Constant(0.0),
            name=f"{self.name}_dense_2" if self.name else None,
        )

        self.act_fn = tf.keras.activations.get(self.activation)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.dense_1(inputs, training=training)

        if self.use_residual:
            x = self.dense_2(x, training=training)
            x = x + inputs

        return self.act_fn(x)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_hidden": self.num_hidden,
                "activation": self.activation,
                "initializer": self.initializer,
                "use_bias": self.use_bias,
                "use_residual": self.use_residual,
                "dtype": str(self.dtype),
            }
        )
        return config

