# Add to your config YAML to control checkpoint saving frequency
# Example usage:
#   optim:
#     epochs: 150
#     lr: 0.001
#     checkpoint_freq: 10  # Save every 10 epochs

from dataclasses import dataclass, field
from typing import Any

@dataclass
class OptimCfg:
    """
    Sub-config for optimizer and learning rate (scheduler).
    """
    epochs: int = 1000
    lr: float = 1e-3
    scheduler: "SchedulerCfg | None" = None
    checkpoint_freq: int = 0  # 0 disables periodic checkpointing
