from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay, LearningRateSchedule
from tensorflow.data import Dataset
from tensorflow.keras.optimizers.schedules import (
    CosineDecay,
    ExponentialDecay,
    LearningRateSchedule,
)

import wandb
from data.dataset import build_dataset
from data.prescribers import build_prescriber
from data.samplers import StereoSampler
from network.global_conformal_model import GlobalConformalModel
from schemas.schemas import ExperimentCfg
from wandb.integration.keras import WandbMetricsLogger

# Default directory for saving model checkpoints
CHECKPOINTS_DIR = Path("checkpoints")


def run() -> None:
    """
    Top-level logic for setting up and running an experiment.
    """
    cfg, _, _ = parse_cfg()

    # Initialize wandb
    wandb_run = None
    if cfg.wandb_project is not None:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            tags=cfg.wandb_tags if cfg.wandb_tags else None,
            config=asdict(cfg),
        )

    # Build training dataset
    ds = build_data(cfg)

    # Build model
    model = build_network(cfg)

    # Build learning rate scheduler (if any)
    lr = build_scheduler(cfg)

    # Compile
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, jit_compile=False)

    # Prepare callbacks
    callbacks = []
    if cfg.wandb_project is not None:
        callbacks.append(WandbMetricsLogger())

    # Add periodic ModelCheckpoint callback if requested
    checkpoint_freq = getattr(cfg.optim, "checkpoint_freq", 0)
    # Determine subfolder for checkpoints
    checkpoint_subdir = None
    if wandb_run is not None:
        checkpoint_subdir = str(CHECKPOINTS_DIR / (wandb_run.name or wandb_run.id))
    else:
        # Find highest run_N in checkpoints/
        import re

        existing = list(CHECKPOINTS_DIR.glob("run_*"))
        max_n = 0
        for d in existing:
            m = re.match(r"run_(\d+)", d.name)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
        checkpoint_subdir = str(CHECKPOINTS_DIR / f"run_{max_n+1}")
    Path(checkpoint_subdir).mkdir(parents=True, exist_ok=True)
    # Only save periodic checkpoints if checkpoint_freq is not None and > 0
    if checkpoint_freq is not None and checkpoint_freq > 0:
        # Save every N epochs using a custom callback
        from keras.callbacks import Callback

        class PeriodicModelCheckpoint(Callback):
            def __init__(self, save_dir, freq):
                super().__init__()
                self.save_dir = Path(save_dir)
                self.freq = freq

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.freq == 0:
                    path = self.save_dir / f"model_epoch{epoch+1:03d}.keras"
                    self.model.save(path)
                    print(f"Saved periodic checkpoint: {path}")

        callbacks.append(PeriodicModelCheckpoint(checkpoint_subdir, checkpoint_freq))

    # Run training
    model.fit(ds, epochs=int(cfg.optim.epochs), callbacks=callbacks, shuffle=True)

    # Save final model (always as 'final_model.keras')
    final_checkpoint_path = Path(checkpoint_subdir) / "final_model.keras"
    model.save(final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")
    
    # Log checkpoint as wandb artifact if using wandb
    if wandb_run is not None:
        artifact = wandb.Artifact(f"model-{wandb_run.id}", type="model")
        artifact.add_file(str(final_checkpoint_path))
        wandb_run.log_artifact(artifact)
        print(f"Checkpoint logged to wandb as artifact: model-{wandb_run.id}")

    # Finish wandb run
    if wandb_run is not None:
        wandb.finish()


def build_data(cfg: ExperimentCfg) -> Dataset:
    keys_cfg = cfg.keys
    data_cfg = cfg.data

    # Build prescriber
    if isinstance(data_cfg.prescribed_R, str):
        prescriber = build_prescriber(data_cfg.prescribed_R)
    else:
        prescriber = build_prescriber(
            data_cfg.prescribed_R.kind,
            **data_cfg.prescribed_R.kwargs,
        )

    # Build sampler
    train_sampler = StereoSampler(
        num_patches=data_cfg.num_patches,
        num_samples=data_cfg.num_samples,
        radial_offset=data_cfg.radial_offset,
    )

    # Build training dataset
    train_dataset = build_dataset(
        patch_coords_key=keys_cfg.patch_coords_key,
        label_key=keys_cfg.label_key,
        normalization_key=keys_cfg.normalization_key,
        sampler=train_sampler,
        prescriber=prescriber,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )

    return train_dataset


def build_network(cfg: ExperimentCfg) -> Model:
    return GlobalConformalModel(cfg)


def build_scheduler(cfg: ExperimentCfg) -> LearningRateSchedule | float:
    """
    Build a learning rate scheduler from configuration.

    If no scheduler is specified, returns a constant learning rate.
    Supports exponential decay and cosine annealing.

    Args:
        cfg: experiment configuration

    Returns:
        LearningRateSchedule | float: learning rate schedule or constant value
    """
    optim_cfg = cfg.optim

    if optim_cfg.scheduler is None:
        return optim_cfg.lr

    scheduler_cfg = optim_cfg.scheduler

    if scheduler_cfg.kind == "exponential_decay":
        return ExponentialDecay(
            initial_learning_rate=optim_cfg.lr,
            **scheduler_cfg.kwargs,
        )

    elif scheduler_cfg.kind == "cosine_annealing":
        # Typically requires at least `decay_steps`
        return CosineDecay(
            initial_learning_rate=optim_cfg.lr,
            **scheduler_cfg.kwargs,
        )

    else:
        raise ValueError(f"Unknown scheduler kind: {scheduler_cfg.kind}")


def parse_cfg() -> tuple[ExperimentCfg, ArgumentParser, dict]:
    """
    Parse YAML/CLI into a typed ExperimentCfg.
    """
    parser = build_parser()
    namespace = parser.parse_args()
    namespace = parser.instantiate_classes(namespace)
    return namespace.cfg, parser, vars(namespace)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="run")
    # Allow for config file to be passed
    parser.add_argument("--hps", action=ActionConfigFile)
    # Expose all config fields under a single root key `cfg`
    parser.add_class_arguments(ExperimentCfg, nested_key="cfg")
    return parser


if __name__ == "__main__":
    run()
