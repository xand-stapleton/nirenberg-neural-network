"""
Spherical Harmonics Expansion Optimization Script (Direct Method)

This script optimizes coefficients of a spherical harmonics expansion to match
a prescribed scalar curvature function. Instead of using a neural network like
the main run.py script, this directly optimizes the coefficients of a finite
spherical harmonics basis by computing the scalar curvature from first principles.

Usage:
    python sh_expansion/run_sh_direct.py --config configs/known/round.yaml
    python sh_expansion/run_sh_direct.py --config configs/known/round.yaml --l_max 6
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prescribers import build_prescriber
from schemas.schemas import ExperimentCfg

# Import shared functions
from sh_functions import (
    SphericalHarmonicsExpansion,
    compute_laplace_beltrami_tf,
    compute_scalar_curvature,
    generate_samples,
    evaluate_prescriber_tf,
    find_next_run_number,
    DEFAULT_L_MAX,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_EPOCHS_MODEL,
    DEFAULT_LR,
    DEFAULT_LR_END,
    REGULARIZATION_WEIGHT,
    EPSILON_COEFF_DISPLAY,
)


def run_optimization(cfg: ExperimentCfg, l_max: int = 4, config_path: str = "unknown", lr: float = 0.01, num_samples: int = 5000, epochs: int = 300) -> Dict:
    """
    Main optimization loop for spherical harmonics expansion.
    
    Args:
        cfg: experiment configuration (used only for prescriber)
        l_max: maximum degree of spherical harmonics expansion
        lr: learning rate
        num_samples: number of samples per patch
        epochs: number of optimization epochs
        
    Returns:
        dict: optimization results including final coefficients and loss
    """
    # Set random seed
    if cfg.seed is not None:
        tf.random.set_seed(cfg.seed)
        np.random.seed(cfg.seed)
    
    # Setup dtype
    dtype = tf.float64 if cfg.dtype == "float64" else tf.float32
    
    print(f"Using TensorFlow with dtype: {dtype}")
    
    # Build prescriber (from existing TensorFlow code)
    data_cfg = cfg.data
    if isinstance(data_cfg.prescribed_R, str):
        prescriber = build_prescriber(data_cfg.prescribed_R)
    else:
        prescriber = build_prescriber(
            data_cfg.prescribed_R.kind,
            **data_cfg.prescribed_R.kwargs,
        )
    
    print(f"Prescriber: {prescriber}")
    
    # Create spherical harmonics expansion model
    u_model = SphericalHarmonicsExpansion(
        max_degree=l_max,
        dtype=dtype
    )

    # Use learning rate from argument (ignore config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Setup learning rate scheduler if specified
    lr_schedule = None
    if cfg.optim.scheduler is not None:
        scheduler_cfg = cfg.optim.scheduler
        if scheduler_cfg.kind == "exponential_decay":
            # Use TensorFlow ExponentialDecay
            decay_rate = scheduler_cfg.kwargs.get("decay_rate", 0.96)
            decay_steps = scheduler_cfg.kwargs.get("decay_steps", 1000)
            staircase = scheduler_cfg.kwargs.get("staircase", False)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=staircase
            )
            # Recreate optimizer with schedule
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            print(f"Using ExponentialDecay scheduler: decay_steps={decay_steps}, decay_rate={decay_rate}")
        elif scheduler_cfg.kind == "cosine_annealing":
            decay_steps = scheduler_cfg.kwargs.get("decay_steps", 1000)
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=decay_steps,
                alpha=DEFAULT_LR_END / lr  # alpha is the final LR as fraction of initial
            )
            # Recreate optimizer with schedule
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            print(f"Using CosineDecay scheduler: decay_steps={decay_steps}, final_lr={DEFAULT_LR_END}")
    
    # Generate samples once (could be regenerated each epoch for variety)
    print(f"Generating {num_samples} samples per patch...")
    sample_coords = generate_samples(
        num_samples=num_samples,
        num_patches=data_cfg.num_patches,
        radial_offset=data_cfg.radial_offset,
        dtype=dtype,
    )
    
    # Compute prescribed R for all samples
    print("Computing prescribed scalar curvature...")
    prescribed_R_list = []
    for patch_idx in range(data_cfg.num_patches):
        coords = sample_coords[patch_idx]  # (num_samples, 2)
        prescribed_R = evaluate_prescriber_tf(prescriber, coords, patch_idx)
        prescribed_R_list.append(prescribed_R)
    prescribed_R_all = tf.concat(prescribed_R_list, axis=0)  # (num_patches * num_samples,)
    
    # Compute normalization constant (for loss scaling)
    normalizer = tf.math.reduce_variance(prescribed_R_all)
    if normalizer < 1e-6:
        normalizer = tf.constant(1.0, dtype=dtype)
    print(f"Normalizer: {normalizer.numpy():.6f}")
    
    # Training loop
    print(f"\nStarting optimization for {epochs} epochs...")
    print("=" * 80)
    
    best_loss = float("inf")
    best_coefficients = None
    
    for epoch in range(epochs):
        # Forward pass with gradient tape
        with tf.GradientTape() as tape:
            # Compute loss for all patches
            predicted_R_list = []
            
            for patch_idx in range(data_cfg.num_patches):
                coords = sample_coords[patch_idx]  # (num_samples, 2)
                
                # Compute u and Laplace-Beltrami
                u, delta_u = compute_laplace_beltrami_tf(u_model, coords, patch_idx)
                
                # Compute predicted scalar curvature
                R_pred = compute_scalar_curvature(u, delta_u)
                predicted_R_list.append(R_pred)
            
            # Concatenate predictions from all patches
            predicted_R_all = tf.concat(predicted_R_list, axis=0)  # (num_patches * num_samples,)
            
            # Compute MSE loss (matching loss/losses.py ScalarLoss)
            mse = tf.reduce_mean((predicted_R_all - prescribed_R_all) ** 2 / normalizer)
            
            # Add L2 regularization on coefficients
            l2_reg = REGULARIZATION_WEIGHT * tf.reduce_sum(u_model.coefficients ** 2)
            
            total_loss = mse + l2_reg
        
        # Backpropagation
        gradients = tape.gradient(total_loss, u_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
        
        # Track best model
        loss_value = total_loss.numpy()
        if loss_value < best_loss:
            best_loss = loss_value
            best_coefficients = tf.identity(u_model.coefficients)
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = optimizer.learning_rate(optimizer.iterations).numpy()
            else:
                current_lr = optimizer.learning_rate.numpy()
            # Compute individual loss components for logging
            mse_val = tf.reduce_mean((predicted_R_all - prescribed_R_all) ** 2 / normalizer).numpy()
            l2_val = (REGULARIZATION_WEIGHT * tf.reduce_sum(u_model.coefficients ** 2)).numpy()
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Total Loss: {loss_value:.6e} | "
                  f"MSE: {mse_val:.6e} | "
                  f"L2: {l2_val:.6e} | "
                  f"LR: {current_lr:.6e}")
    
    print("=" * 80)
    print(f"Optimization complete! Best loss: {best_loss:.6e}")
    
    # Restore best coefficients
    u_model.coefficients.assign(best_coefficients)
    
    # Save results
    results = {
        "coefficients": {},
        "best_loss": best_loss,
        "l_max": l_max,
        "num_coefficients": len(u_model.lm_pairs),
    }
    
    # Store coefficients with their (l, m) indices
    print("\nFinal coefficients (c_{l,m} values - used as c/(l(l+1)) in ansatz):")
    print("Note: For SpectralPair (prop) prescribers with coefficient c_prescriber,")
    print("      the learned c_{l,m} should equal c_prescriber (not c_prescriber/(l(l+1)))")
    
    coeffs_numpy = u_model.coefficients.numpy()
    max_coeff = np.max(np.abs(coeffs_numpy))
    threshold = max_coeff * EPSILON_COEFF_DISPLAY
    
    print(f"Significant coefficients (|c| > {threshold:.6e}, i.e., {EPSILON_COEFF_DISPLAY*100:.1f}% of max):")
    print("-" * 60)
    for idx, (l, m) in enumerate(u_model.lm_pairs):
        coeff_value = coeffs_numpy[idx]
        l_val = u_model.lm_pairs[idx][0]
        results["coefficients"][f"c_{l}_{m}"] = float(coeff_value)
        if abs(coeff_value) > threshold:
            # Show both the raw coefficient and the effective coefficient in u
            eff_coeff = coeff_value / (l_val * (l_val + 1))
            print(f"  c({l:2d}, {m:3d}) = {coeff_value:12.6e}  [effective in u: {eff_coeff:12.6e}]")
    print("-" * 60)
    
    # Save coefficients to file
    output_dir = Path(__file__).parent.parent / "results" / "sh_expansion"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next run number
    next_run = find_next_run_number(output_dir)
    output_file = output_dir / f"coefficients_run{next_run}.npz"
    
    # Get prescriber name - try to use a more descriptive identifier
    if isinstance(data_cfg.prescribed_R, str):
        prescriber_name = data_cfg.prescribed_R
        prescriber_kwargs = {}
    else:
        prescriber_name = data_cfg.prescribed_R.kind
        prescriber_kwargs = data_cfg.prescribed_R.kwargs if hasattr(data_cfg.prescribed_R, 'kwargs') else {}
        # Add kwargs info if they exist to make it more descriptive
        if prescriber_kwargs:
            # Create a simple descriptor from kwargs
            kwargs_items = []
            for k, v in prescriber_kwargs.items():
                if isinstance(v, list):
                    kwargs_items.append(f"{k}={','.join(map(str, v))}")
                else:
                    kwargs_items.append(f"{k}={v}")
            if kwargs_items:
                prescriber_name = f"{prescriber_name}({';'.join(kwargs_items)})"
    
    # Build save dict with basic metadata
    save_dict = {
        'coefficients': coeffs_numpy,
        'lm_pairs': np.array(u_model.lm_pairs),
        'loss': best_loss,
        'l_max': l_max,
        # Run metadata
        'prescriber': prescriber_name,
        'config_file': config_path,
        'num_samples': num_samples,
        'num_patches': data_cfg.num_patches,
        'epochs': epochs,
        'learning_rate': lr,
        'scheduler': cfg.optim.scheduler.kind if cfg.optim.scheduler else "none",
        'dtype': cfg.dtype,
    }
    
    # For SpectralPair (prop) prescribers, save the original parameters for exact reproduction
    if isinstance(data_cfg.prescribed_R, str):
        kind = data_cfg.prescribed_R
    else:
        kind = data_cfg.prescribed_R.kind
    
    if kind == 'prop' and prescriber_kwargs:
        # Save the original SpectralPair parameters
        if 'ls' in prescriber_kwargs:
            save_dict['original_prop_ls'] = np.array(prescriber_kwargs['ls'])
        if 'cs' in prescriber_kwargs:
            save_dict['original_prop_cs'] = np.array(prescriber_kwargs['cs'])
        if 'ms' in prescriber_kwargs:
            save_dict['original_prop_ms'] = np.array(prescriber_kwargs['ms'])
    
    np.savez(output_file, **save_dict)
    print(f"\nCoefficients saved to: {output_file}")
    
    return results


def parse_cfg() -> tuple[ExperimentCfg, int, str, float, int, int]:
    """
    Parse YAML/CLI into a typed ExperimentCfg.
    
    Returns:
        cfg: Experiment configuration (used only for prescriber)
        l_max: Maximum degree of spherical harmonics
        config_path: Path to config file (or "unknown" if not provided)
        lr: Learning rate
        num_samples: Number of samples per patch
        epochs: Number of optimization epochs
    """
    parser = argparse.ArgumentParser(
        prog="run_sh_direct",
        description="Spherical Harmonics Expansion Optimization (Direct Method)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--l_max",
        type=int,
        default=DEFAULT_L_MAX,
        help=f"Maximum degree of spherical harmonics expansion"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples to generate per patch"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS_MODEL,
        help="Number of optimization epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for Adam optimizer (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Load YAML config
    config_path = args.config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Handle configs with 'cfg:' wrapper (like those used by main run.py)
    if 'cfg' in config_data:
        config_dict = config_data['cfg']
    else:
        config_dict = config_data
    
    # Convert dict to ExperimentCfg
    # Note: This is a simplified version that assumes the YAML structure matches ExperimentCfg
    # For more complex configs, you might need dacite or similar
    from schemas.schemas import DataCfg, NetworkCfg, LossCfg, OptimCfg, PrescriberCfg, SchedulerCfg
    
    # Helper to build nested configs
    def build_config(config_class, config_dict):
        if config_dict is None:
            return config_class()
        kwargs = {}
        for field_name, field_value in config_dict.items():
            kwargs[field_name] = field_value
        return config_class(**kwargs)
    
    # Build data config
    data_dict = config_dict.get('data', {})
    if 'prescribed_R' in data_dict and isinstance(data_dict['prescribed_R'], dict):
        data_dict['prescribed_R'] = PrescriberCfg(**data_dict['prescribed_R'])
    data_cfg = build_config(DataCfg, data_dict)
    
    # Build network config
    network_cfg = build_config(NetworkCfg, config_dict.get('network', {}))
    
    # Build loss config
    loss_cfg = build_config(LossCfg, config_dict.get('loss', {}))
    
    # Build optimizer config
    optim_dict = config_dict.get('optim', {})
    if 'scheduler' in optim_dict and optim_dict['scheduler'] is not None:
        optim_dict['scheduler'] = SchedulerCfg(**optim_dict['scheduler'])
    optim_cfg = build_config(OptimCfg, optim_dict)
    
    # Build top-level config
    cfg = ExperimentCfg(
        data=data_cfg,
        network=network_cfg,
        loss=loss_cfg,
        optim=optim_cfg,
        seed=config_dict.get('seed', 42),
        dtype=config_dict.get('dtype', 'float64'),
        wandb_project=config_dict.get('wandb_project'),
        wandb_entity=config_dict.get('wandb_entity'),
        wandb_name=config_dict.get('wandb_name'),
        wandb_tags=config_dict.get('wandb_tags', []),
        checkpoint_name=config_dict.get('checkpoint_name'),
    )
    
    return cfg, args.l_max, config_path, args.lr, args.num_samples, args.epochs


def main():
    """
    Main entry point.
    """
    print("=" * 80)
    print("Spherical Harmonics Expansion Optimization (Direct Method)")
    print("=" * 80)
    print()
    
    # Parse configuration
    cfg, l_max, config_path, lr, num_samples, epochs = parse_cfg()

    print(f"Config file: {config_path}")
    print(f"l_max = {l_max}")
    print(f"Learning rate: {lr}")
    print(f"Number of samples: {num_samples}")
    print(f"Epochs: {epochs}")
    print()

    # Run optimization
    results = run_optimization(cfg, l_max, config_path, lr, num_samples, epochs)

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
