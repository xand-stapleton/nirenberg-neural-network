import tensorflow as tf
from keras import Model
from keras.initializers import Constant, RandomNormal
from keras.layers import Dense, Input
from keras.metrics import Mean
from keras.saving import register_keras_serializable

from geometry.ball import analytic_round_metric, patch_xy_to_xyz
from loss.losses import CompoundLoss, LossTerm, ScalarLoss
from network.resblock import ResidualBlock
from network.rff import RandomFourierFeature
from schemas.schemas import ExperimentCfg

@register_keras_serializable(package="Custom", name="GlobalConformalModel")
class GlobalConformalModel(Model):
    """
    PINN for the Kazdan-Warner problem on the sphere.

    Args:
        cfg: hyperparameters config

    """

    def __init__(self, cfg: ExperimentCfg):
        super().__init__(dtype=cfg.dtype)

        self.cfg = cfg

        keys_cfg = self.cfg.keys
        data_cfg = self.cfg.data
        network_cfg = self.cfg.network

        # Batch dict keys
        self.patch_coords_key = keys_cfg.patch_coords_key
        self.label_key = keys_cfg.label_key
        self.conformal_factor_key = keys_cfg.conformal_factor_key
        self.conformal_metric_key = keys_cfg.conformal_metric_key
        self.laplace_beltrami_key = keys_cfg.laplace_beltrami_key
        self.normalization_key = keys_cfg.normalization_key

        # Data-related hyperparameters
        self.num_patches = data_cfg.num_patches

        # Network-related hyperparameters
        self.num_hidden = network_cfg.num_hidden
        self.num_layers = network_cfg.num_layers
        self.activation = network_cfg.activation
        self.initializer = network_cfg.initializer
        self.use_bias = network_cfg.use_bias
        self.use_residual = network_cfg.use_residual
        self.use_rffs = network_cfg.use_rffs
        self.num_rffs = network_cfg.num_rffs
        self.rff_sigma = network_cfg.rff_sigma

        # Save loss config
        self.loss_cfg = cfg.loss
        self.loss_tracker = Mean(name=self.loss_cfg.name)

        # Build smooth neural surrogate for conformal factor
        self.u_model = self._build_u_model()

        # Build loss
        self.compound_loss = self._build_loss()

    def call(self, batch_dict: dict, training: bool = False) -> dict:
        # Extract model input key (patch coordinates)
        vw = batch_dict[self.patch_coords_key]

        us, delta_us, gs = [], [], []
        for patch_idx in range(self.num_patches):
            # Extract current patch
            vw_idx = vw[:, patch_idx, :]
            # Compute u and Laplace-Beltrami of u w.r.t. round metric
            u, delta_u = self._compute_laplace_beltrami(
                vw_idx,
                patch_idx,
                training,
            )
            # Compute predicted metric
            g = self._compute_global_conformal_metric(u, vw_idx)
            # Append
            us.append(u)
            delta_us.append(delta_u)
            gs.append(g)

        # Stack features that were computed per patch
        u = tf.stack(us, axis=1)
        delta_u = tf.stack(delta_us, axis=1)
        g = tf.stack(gs, axis=1)

        # Store features to batch dict
        batch_dict[self.conformal_factor_key] = u
        batch_dict[self.laplace_beltrami_key] = delta_u
        batch_dict[self.conformal_metric_key] = g

        return batch_dict

    def train_step(self, batch_dict: dict) -> dict:
        """
        Apply forward pass, compute loss, and backpropagate.
        """
        with tf.GradientTape() as tape:
            batch_dict = self(batch_dict, training=True)
            loss = self._compute_loss(batch_dict)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Compute metrics
        self.loss_tracker.update_state(loss)

        return {self.loss_cfg.name: self.loss_tracker.result()}

    @property
    def metrics(self) -> list:
        # We list our `Metric` objects here so that `reset_states()` can be called
        # automatically at the start of each epoch or at the start of `evaluate()`.
        return [self.loss_tracker]

    def predict_u(self, patch_coords: tf.Tensor, patch_idx: int = 0) -> tf.Tensor:
        """
        Predict the conformal factor u at input points.
        
        Args:
            patch_coords: (B, 2) or (N, num_patches, 2)
                Chart coordinates. If shape is (B, 2), uses patch_idx.
                If shape is (N, num_patches, 2), predicts for all patches.
            patch_idx: int, default=0
                Which patch to use (0=north, 1=south). Ignored if patch_coords
                has shape (N, num_patches, 2).
        
        Returns:
            tf.Tensor: (B, 1) or (N, num_patches, 1)
                Conformal factor u evaluated at input points.
        """
        # Check input shape
        if len(patch_coords.shape) == 3:
            # Shape (N, num_patches, 2) - predict for all patches
            us = []
            for idx in range(self.num_patches):
                vw_idx = patch_coords[:, idx, :]
                xyz = patch_xy_to_xyz(vw_idx, idx)
                u = self.u_model(xyz, training=False)
                us.append(u)
            return tf.stack(us, axis=1)
        else:
            # Shape (B, 2) - predict for single patch
            xyz = patch_xy_to_xyz(patch_coords, patch_idx)
            return self.u_model(xyz, training=False)
    
    def predict_g(self, patch_coords: tf.Tensor, patch_idx: int = 0) -> tf.Tensor:
        """
        Predict the conformal metric g = e^(2u) * g0 at input points.
        
        Args:
            patch_coords: (B, 2) or (N, num_patches, 2)
                Chart coordinates. If shape is (B, 2), uses patch_idx.
                If shape is (N, num_patches, 2), predicts for all patches.
            patch_idx: int, default=0
                Which patch to use (0=north, 1=south). Ignored if patch_coords
                has shape (N, num_patches, 2).
        
        Returns:
            tf.Tensor: (B, 2, 2) or (N, num_patches, 2, 2)
                Conformal metric g evaluated at input points.
        """
        # Check input shape
        if len(patch_coords.shape) == 3:
            # Shape (N, num_patches, 2) - predict for all patches
            gs = []
            for idx in range(self.num_patches):
                vw_idx = patch_coords[:, idx, :]
                xyz = patch_xy_to_xyz(vw_idx, idx)
                u = self.u_model(xyz, training=False)
                g = self._compute_global_conformal_metric(u, vw_idx)
                gs.append(g)
            return tf.stack(gs, axis=1)
        else:
            # Shape (B, 2) - predict for single patch
            xyz = patch_xy_to_xyz(patch_coords, patch_idx)
            u = self.u_model(xyz, training=False)
            return self._compute_global_conformal_metric(u, patch_coords)

    def _compute_loss(self, batch_dict: dict) -> tf.Tensor:
        batch_dict = self.compound_loss(batch_dict)
        return batch_dict[self.compound_loss.name]

    def _build_u_model(self) -> Model:
        """
        Build the neural surrogate (MLP) for modeling the conformal factor u.

        Returns:
            Model

        """
        # Input dimension is hardcoded to 3, for now
        inp = Input(shape=(3,), dtype=self.dtype)
        x = inp

        # Optionally encode the input using random Fourier features
        if self.use_rffs:
            rff = RandomFourierFeature(
                num_features=self.num_rffs,
                sigma=self.rff_sigma,
                dtype=self.dtype,
            )
            x = rff(x)

        # Take care of any input dimension mismatch
        input_dim = 2 * self.num_rffs if self.use_rffs else 3
        if input_dim != self.num_hidden:
            x = Dense(
                units=self.num_hidden,
                activation=self.activation,
                use_bias=self.use_bias,
                dtype=self.dtype,
                kernel_initializer=self.initializer,
            )(x)

        # Initialize all dense layers but the last
        for block_idx in range(self.num_layers - 2):
            residual_block = ResidualBlock(
                num_hidden=self.num_hidden,
                activation=self.activation,
                initializer=self.initializer,
                use_bias=self.use_bias,
                use_residual=self.use_residual,
                dtype=self.dtype,
                name=f"residual_block_{block_idx}",
            )
            x = residual_block(x)

        # Initialize final dense layer with guaranteed bias
        out = Dense(
            units=1,
            activation=None,
            use_bias=True,
            dtype=self.dtype,
            kernel_initializer=RandomNormal(stddev=1e-3),
            bias_initializer=Constant(0.0),
        )(x)

        return Model(inp, out, name="u_model", dtype=self.dtype)

    def _build_loss(self) -> CompoundLoss:
        """
        Build the complete loss function (CompoundLoss) by piecing together different
        terms (LossTerm). At the very least, this yields a ScalarLoss term.

        Returns:
            CompoundLoss

        """
        terms: list[LossTerm] = []
        multipliers: list[float] = []
        scalar_loss_cfg = self.loss_cfg.scalar_loss

        # The parsed ExperimentCfg is guaranteed to have a ScalarLossCfg
        terms.append(
            ScalarLoss(
                name=scalar_loss_cfg.name,
                conformal_factor_key=self.conformal_factor_key,
                laplace_beltrami_key=self.laplace_beltrami_key,
                label_key=self.label_key,
                normalization_key=self.normalization_key,
            )
        )
        multipliers.append(scalar_loss_cfg.multiplier)

        return CompoundLoss(
            name=self.loss_cfg.name,
            terms=terms,
            multipliers=multipliers,
            dtype=self.dtype,
        )

    def _compute_global_conformal_metric(
        self,
        u: tf.Tensor,
        patch_coords: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the conformal metric e^2u * g0.

        Args:
            u: (B, 1)
                conformal factor
            patch_coords: (B, 2)
                chart coordinates

        Returns:
            tf.Tensor: (B, 2, 2)
                conformal metric, evaluated at each batch point

        """
        # Compute conformal factor. Adjust shape for broadcast: (B, 1) -> (B, 1, 1)
        e2u = tf.exp(2.0 * u)[..., None]
        # Compute round metric at patch coordinates: (B, 2, 2)
        g0 = analytic_round_metric(patch_coords)
        return e2u * g0


    @tf.function(reduce_retracing=True)
    def _compute_laplace_beltrami(
        self,
        patch_coords: tf.Tensor,
        patch_idx: int,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Compute u. Compute the Laplace-Beltrami operator Δ_g0 u. This is required
        downstream for the scalar loss.

        Args:
            patch_coords: (B, 2)
                chart coordinates
            patch_idx: 0 (north) or 1 (south)
            training: whether model is in training mode

        Returns:
            tf.Tensor: (B, 1)
                u, evaluated at each batch point
            tf.Tensor: (B, 1)
                Δ_g0 u, evaluated at each batch point

        """
        # Divergence of weighted gradient
        # Note: Tapes must be nested for second-order derivatives to work correctly.
        # persistent=True is required because we need to compute the jacobian
        # of weighted_grad_u, which depends on g0 and grad_u
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(patch_coords)
            # Gradient of u w.r.t. patch coords (computed inside t2's context)
            with tf.GradientTape() as t1:
                t1.watch(patch_coords)
                xyz = patch_xy_to_xyz(patch_coords, patch_idx)
                u = self.u_model(xyz, training=training)
            grad_u = t1.gradient(u, patch_coords)
            # Evaluate (round) metric at patch coords
            g0 = analytic_round_metric(patch_coords)
            # Compute determinant and square root
            det_g0 = tf.linalg.det(g0)
            sqrt_det_g0 = tf.sqrt(det_g0)
            # Compute metric inverse
            g0_inv = tf.linalg.inv(g0)
            # Raise index: grad_u_raised^i = g0^{ij} ∂_j u
            grad_u_raised = tf.einsum("bij, bj -> bi", g0_inv, grad_u)
            # Weight by √|g0|: weighted_grad_u^i = √|g0| * grad_u_raised^i
            weighted_grad_u = sqrt_det_g0[..., None] * grad_u_raised
            
        # Final divergence: div = ∂_i (weighted_grad_u^i)
        J = t2.batch_jacobian(weighted_grad_u, patch_coords)
        div_weighted_grad = J[:, 0, 0] + J[:, 1, 1]

        # Final Laplace–Beltrami: Δu = (1/√|g0|) * div
        # Note: sqrt_det_g0 is still in scope from the tape context above
        delta_u = (div_weighted_grad / sqrt_det_g0)[:, None]
        
        # Clean up persistent tape
        del t2
        
        return u, delta_u

    def get_config(self) -> dict:
        """
        Get model configuration for serialization.

        Returns:
            dict: configuration dictionary

        """
        from dataclasses import asdict

        return {"cfg": asdict(self.cfg)}

    def get_build_config(self) -> dict:
        """
        Get build configuration so Keras can recreate variables on load.

        Returns:
            dict: build configuration dictionary

        """
        # Batch dimension is left dynamic; other dims are fixed by config.
        return {"patch_coords_shape": (None, self.num_patches, 2)}

    def build_from_config(self, config: dict) -> None:
        """
        Build model variables from build configuration.

        Args:
            config: build configuration dictionary

        """
        patch_coords_shape = config.get("patch_coords_shape")
        if not patch_coords_shape:
            return

        batch_dim = patch_coords_shape[0] or 1
        num_patches = patch_coords_shape[1]
        coord_dim = patch_coords_shape[2]

        dummy_patch_coords = tf.zeros(
            (batch_dim, num_patches, coord_dim),
            dtype=self.dtype,
        )
        # Running a forward pass builds all variables, including the submodel.
        _ = self({self.patch_coords_key: dummy_patch_coords}, training=False)

    @classmethod
    def from_config(cls, config: dict) -> "GlobalConformalModel":
        """
        Create model from configuration dictionary.

        Args:
            config: serialized configuration dictionary

        Returns:
            GlobalConformalModel: reconstructed model

        """
        from dacite import from_dict

        from schemas.schemas import ExperimentCfg

        cfg_dict = config["cfg"]
        # Backward-compat: map legacy prescribed_K/prescribed_k to prescribed_R (including nested kind/kwargs)
        data_cfg = cfg_dict.get("data", {})
        if "prescribed_R" not in data_cfg:
            if "prescribed_K" in data_cfg:
                data_cfg["prescribed_R"] = data_cfg.get("prescribed_K")
            elif "prescribed_k" in data_cfg:
                data_cfg["prescribed_R"] = data_cfg.get("prescribed_k")
        if "prescribed_K" in data_cfg:
            data_cfg.pop("prescribed_K", None)
        if "prescribed_k" in data_cfg:
            data_cfg.pop("prescribed_k", None)
        cfg_dict["data"] = data_cfg

        # Backward-compat: update legacy label_key if it referenced prescribed_K
        keys_cfg = cfg_dict.get("keys", {})
        if keys_cfg.get("label_key") == "prescribed_K":
            keys_cfg["label_key"] = "prescribed_R"
        cfg_dict["keys"] = keys_cfg

        cfg = from_dict(data_class=ExperimentCfg, data=cfg_dict)
        return cls(cfg)
