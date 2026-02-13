from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentCfg:
    """
    Top-level config.
    """

    keys: "KeysCfg" = field(default_factory=lambda: KeysCfg())
    data: "DataCfg" = field(default_factory=lambda: DataCfg())
    network: "NetworkCfg" = field(default_factory=lambda: NetworkCfg())
    loss: "LossCfg" = field(default_factory=lambda: LossCfg())
    optim: "OptimCfg" = field(default_factory=lambda: OptimCfg())
    seed: float | None = 42
    dtype: str = "float64"

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    checkpoint_name: str | None = None


@dataclass
class KeysCfg:
    """
    Sub-config for batch dictionary keys.

    Notes:
        - This sub-config should not require any alteration or overriding in most use cases.

    """

    patch_coords_key: str = "patch_coords"
    label_key: str = "prescribed_R"
    conformal_factor_key: str = "u"
    conformal_metric_key: str = "g"
    laplace_beltrami_key: str = "delta_u"
    normalization_key: str = "normalizer"


@dataclass
class DataCfg:
    """
    Sub-config for data generation.
    """

    num_patches: int = 2
    num_samples: int = 10_000
    expected_existence: bool | None = None
    radial_offset: float = 0.01
    prescribed_R: "PrescriberCfg | str" = "round"
    batch_size: int = 100


@dataclass
class PrescriberCfg:
    """
    Sub-config for prescribed scalar curvature function.

    Notes:
        - This must be used if the user wishes to pass keyword arguments to the prescriber.

    """

    kind: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkCfg:
    """
    Sub-config for network hyperparameters.
    """

    num_hidden: int = 64
    num_layers: int = 3
    activation: str = "gelu"
    initializer: str = "he_uniform"
    use_bias: bool = True
    use_residual: bool = True

    use_rffs: bool = True
    num_rffs: int = 16
    rff_sigma: float = 1.0


@dataclass
class LossCfg:
    """
    Sub-config for compound loss.

    Notes:
        - Currently only supports a scalar loss term.

    """

    scalar_loss: "ScalarLossCfg | None" = None
    name: str = "conformal_loss"


@dataclass
class ScalarLossCfg:
    """
    Sub-config for a scalar loss term.
    """

    multiplier: float = 1.0
    name: str = "scalar_loss"


@dataclass
class OptimCfg:
    """
    Sub-config for optimizer and learning rate (scheduler).
    """

    epochs: int = 1000
    lr: float = 1e-3
    scheduler: "SchedulerCfg | None" = None
    checkpoint_freq: int | None = 0  # Save model every N epochs (0 or None disables periodic checkpointing)


@dataclass
class SchedulerCfg:
    """
    Sub-config for learning rate scheduler.
    """

    kind: str = "exponential_decay"
    kwargs: dict[str, Any] = field(default_factory=dict)
