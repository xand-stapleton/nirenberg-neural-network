"""
Spherical Harmonics Expansion Optimization Script (Model-Based Method)

This script fits spherical harmonics coefficients to match the conformal factor
predicted by a trained neural network model. This allows extracting an analytical
spherical harmonics representation from a learned neural network solution.

Usage:
    python sh_expansion/run_sh_model.py --model clean-dust-3494 --l_max 6
    python sh_expansion/run_sh_model.py --model clean-dust-3494 --l_max 4 --num_samples 20000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.global_conformal_model import GlobalConformalModel

# Import shared functions
from sh_functions import (
    SphericalHarmonicsExpansion,
    generate_samples,
    xyz_to_spherical,
    find_next_run_number,
    add_common_args,
    DEFAULT_L_MAX,
    DEFAULT_MODEL_CHECKPOINT,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_NUM_PATCHES,
    DEFAULT_EPOCHS_MODEL,
    DEFAULT_LR,
    DEFAULT_LR_END,
    DEFAULT_DTYPE,
    DEFAULT_SEED,
    REGULARIZATION_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    EPSILON_COEFF_DISPLAY,
)
from geometry.ball import patch_xy_to_xyz


# ============================================================================
# Main Functions
# ============================================================================

def load_model(checkpoint_name: str) -> GlobalConformalModel:
    """
    Load a trained model from checkpoints directory.
    
    Args:
        checkpoint_name: name of checkpoint folder (e.g., "clean-dust-3494") 
                        or full path to model file
        
    Returns:
        GlobalConformalModel: loaded model
    """
    # Handle both full paths and checkpoint names
    checkpoint_path = Path(checkpoint_name)
    if checkpoint_path.suffix == '.keras':
        # Full path to model file provided
        model_path = checkpoint_path
    elif '/' in checkpoint_name or checkpoint_path.is_absolute():
        # Path to checkpoint directory provided
        model_path = checkpoint_path / "final_model.keras"
    else:
        # Just checkpoint name provided
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / checkpoint_name
        model_path = checkpoint_dir / "final_model.keras"
    
    if not model_path.exists():
        checkpoint_dir = model_path.parent
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Available checkpoints in {checkpoint_dir.parent}:\n" +
            "\n".join(f"  - {p.name}" for p in checkpoint_dir.parent.iterdir() if p.is_dir() and not p.name.startswith("."))
        )
    
    print(f"Loading model from: {model_path}")
    
    # Load the Keras model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Read config from model (automatically saved via get_config/from_config)
    if hasattr(model, 'cfg'):
        print(f"Model configuration loaded:")
        print(f"  prescribed_R: {model.cfg.data.prescribed_R}")
    else:
        print("Warning: Model has no cfg attribute")
    
    return model


def evaluate_model_on_sphere(
    model: GlobalConformalModel,
    sample_coords: tf.Tensor,
    dtype: tf.DType,
) -> tf.Tensor:
    """
    Evaluate the trained model's conformal factor u on sphere samples.
    
    Args:
        model: trained GlobalConformalModel
        sample_coords: (num_patches, num_samples, 2) patch coordinates
        dtype: tf data type
        
    Returns:
        tf.Tensor: (num_patches * num_samples,) predicted u values
    """
    num_patches = sample_coords.shape[0]
    num_samples = sample_coords.shape[1]
    
    # Reshape to (num_samples, num_patches, 2) as expected by GlobalConformalModel
    # sample_coords is currently (num_patches, num_samples, 2)
    coords_reshaped = tf.transpose(sample_coords, [1, 0, 2])  # (num_samples, num_patches, 2)
    
    # Create batch dict as expected by model
    batch_dict = {
        'patch_coords': coords_reshaped
    }
    
    # Evaluate model - returns dict with 'u' key
    output = model(batch_dict, training=False)
    
    # Extract u values: shape is (num_samples, num_patches, 1)
    u_pred = output['u']  # (num_samples, num_patches, 1)
    
    # Reshape to (num_patches, num_samples) then flatten
    u_pred = tf.squeeze(u_pred, axis=-1)  # (num_samples, num_patches)
    u_pred = tf.transpose(u_pred, [1, 0])  # (num_patches, num_samples)
    u_pred = tf.reshape(u_pred, [-1])  # (num_patches * num_samples,)
    
    return u_pred


def fit_spherical_harmonics(
    target_u: tf.Tensor,
    sample_coords: tf.Tensor,
    num_patches: int,
    l_max: int,
    epochs: int,
    learning_rate: float,
    dtype: tf.DType,
) -> tuple[tf.Tensor, np.ndarray, float]:
    """
    Fit spherical harmonics coefficients to match target u values.
    
    Args:
        target_u: (num_patches * num_samples,) target u values from model
        sample_coords: (num_patches, num_samples, 2) patch coordinates
        num_patches: number of patches
        l_max: maximum degree of SH expansion
        epochs: number of optimization epochs
        learning_rate: learning rate for Adam optimizer
        dtype: tf data type
        
    Returns:
        coefficients: fitted SH coefficients
        lm_pairs: corresponding (l, m) indices
        loss: final MSE loss
    """
    # Create spherical harmonics expansion model
    u_model = SphericalHarmonicsExpansion(max_degree=l_max, dtype=dtype)
    
    # Setup learning rate schedule (cosine annealing works well for fitting)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs,
        alpha=DEFAULT_LR_END / learning_rate  # alpha is the final LR as fraction of initial
    )
    
    # Setup optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Precompute spherical coordinates for all samples
    theta_list = []
    phi_list = []
    
    for patch_idx in range(num_patches):
        coords = sample_coords[patch_idx]  # (num_samples, 2)
        xyz = patch_xy_to_xyz(coords, patch_idx)
        theta, phi = xyz_to_spherical(xyz)
        theta_list.append(theta)
        phi_list.append(phi)
    
    theta_all = tf.concat(theta_list, axis=0)  # (num_patches * num_samples,)
    phi_all = tf.concat(phi_list, axis=0)
    
    print(f"\nFitting spherical harmonics to model predictions...")
    print(f"Target u statistics: mean={tf.reduce_mean(target_u):.6f}, "
          f"std={tf.math.reduce_std(target_u):.6f}")
    print(f"Using cosine annealing schedule and early stopping (patience={EARLY_STOPPING_PATIENCE})")
    print("=" * 80)
    
    best_loss = float("inf")
    best_coefficients = None
    patience_counter = 0
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Predict u using SH expansion
            u_pred = u_model(theta_all, phi_all)
            
            # Compute MSE loss
            mse = tf.reduce_mean((u_pred - target_u) ** 2)
            
            # Add L2 regularization on coefficients
            l2_reg = REGULARIZATION_WEIGHT * tf.reduce_sum(u_model.coefficients ** 2)
            
            total_loss = mse + l2_reg
        
        # Backpropagation
        gradients = tape.gradient(total_loss, u_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
        
        # Track best model and early stopping
        loss_value = total_loss.numpy()
        # Only consider it an improvement if loss decreases by at least this relative amount
        improvement_threshold = 1e-6
        if loss_value < best_loss * (1 - improvement_threshold):
            best_loss = loss_value
            best_coefficients = tf.identity(u_model.coefficients)
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Compute individual loss components for logging
            mse_val = mse.numpy()
            l2_val = l2_reg.numpy()
            current_lr = lr_schedule(optimizer.iterations).numpy()
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Total Loss: {loss_value:.6e} | "
                  f"MSE: {mse_val:.6e} | "
                  f"L2: {l2_val:.6e} | "
                  f"LR: {current_lr:.6e}")
        
        # Early stopping check
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
    
    print("=" * 80)
    print(f"Fitting complete! Best loss: {best_loss:.6e}")
    
    # Restore best coefficients
    u_model.coefficients.assign(best_coefficients)
    
    return u_model.coefficients.numpy(), np.array(u_model.lm_pairs), best_loss


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fit spherical harmonics coefficients to trained model predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT,
        help="Model checkpoint folder name in checkpoints/"
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
    parser.add_argument(
        "--l_max",
        type=int,
        default=DEFAULT_L_MAX,
        help=f"Maximum degree of spherical harmonics expansion (default: {DEFAULT_L_MAX})"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    print("=" * 80)
    print("Spherical Harmonics Expansion Optimization (Model-Based Method)")
    print("=" * 80)
    print()
    # Use hardcoded defaults for dtype, seed, num_patches
    dtype = tf.float64 if DEFAULT_DTYPE == "float64" else tf.float32
    num_patches = DEFAULT_NUM_PATCHES
    
    print(f"Model checkpoint: {args.model}")
    print(f"l_max: {args.l_max}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of patches: {num_patches}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Dtype: {DEFAULT_DTYPE}")
    print(f"Seed: {DEFAULT_SEED}")
    print()
    
    # Set random seed
    tf.random.set_seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    
    # Load trained model
    model = load_model(args.model)
    print()
    
    # Generate samples on the sphere
    print(f"Generating {args.num_samples} samples per patch...")
    sample_coords = generate_samples(
        num_samples=args.num_samples,
        num_patches=num_patches,
        radial_offset=0.0,  # Not used for uniform sampling
        dtype=dtype,
    )
    print(f"Sample coordinates shape: {sample_coords.shape}")
    print()
    
    # Evaluate model on these samples
    print("Evaluating model predictions on samples...")
    target_u = evaluate_model_on_sphere(model, sample_coords, dtype)
    print(f"Model predictions shape: {target_u.shape}")
    print()
    
    # Fit spherical harmonics to model predictions
    coefficients, lm_pairs, loss = fit_spherical_harmonics(
        target_u=target_u,
        sample_coords=sample_coords,
        num_patches=num_patches,
        l_max=args.l_max,
        epochs=args.epochs,
        learning_rate=args.lr,
        dtype=dtype,
    )
    print()
    
    # Display significant coefficients
    max_coeff = np.max(np.abs(coefficients))
    threshold = max_coeff * EPSILON_COEFF_DISPLAY
    print(f"Significant coefficients (|c| > {threshold:.6e}, i.e., {EPSILON_COEFF_DISPLAY*100:.1f}% of max):")
    print("-" * 60)
    for idx, (l, m) in enumerate(lm_pairs):
        coeff_value = coefficients[idx]
        if abs(coeff_value) > threshold:
            print(f"  c({l:2d}, {m:3d}) = {coeff_value:12.6e}")
    print("-" * 60)
    print()
    
    # Save coefficients to file
    output_dir = Path(__file__).parent.parent / "results" / "sh_expansion"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next run number
    next_run = find_next_run_number(output_dir)
    output_file = output_dir / f"coefficients_run{next_run}.npz"
    
    # Extract prescribed_R from model.cfg
    prescribed_R = "unknown"
    save_kwargs = {}
    if hasattr(model, 'cfg'):
        prescribed_R_obj = model.cfg.data.prescribed_R
        
        # Handle both string and PrescriberCfg formats
        if isinstance(prescribed_R_obj, str):
            prescribed_R = prescribed_R_obj
        else:
            # It's a PrescriberCfg object
            prescribed_R = prescribed_R_obj.kind
            if hasattr(prescribed_R_obj, 'kwargs') and prescribed_R_obj.kwargs:
                # Create descriptive string
                kwargs_items = []
                for k, v in prescribed_R_obj.kwargs.items():
                    if isinstance(v, list):
                        kwargs_items.append(f"{k}={','.join(map(str, v))}")
                    else:
                        kwargs_items.append(f"{k}={v}")
                if kwargs_items:
                    prescribed_R = f"{prescribed_R}({';'.join(kwargs_items)})"
                
                # For prop prescribers, save raw kwargs for exact reconstruction
                if prescribed_R_obj.kind == 'prop':
                    if 'ls' in prescribed_R_obj.kwargs:
                        save_kwargs['original_prop_ls'] = prescribed_R_obj.kwargs['ls']
                    if 'cs' in prescribed_R_obj.kwargs:
                        save_kwargs['original_prop_cs'] = prescribed_R_obj.kwargs['cs']
                    if 'ms' in prescribed_R_obj.kwargs:
                        save_kwargs['original_prop_ms'] = prescribed_R_obj.kwargs['ms']
    
    np.savez(
        output_file,
        coefficients=coefficients,
        lm_pairs=lm_pairs,
        loss=loss,
        l_max=args.l_max,
        # Run metadata
        method="model_based",
        model_checkpoint=args.model,
        prescriber=prescribed_R,  # Add prescriber info from model metadata
        num_samples=args.num_samples,
        num_patches=num_patches,
        epochs=args.epochs,
        learning_rate=args.lr,
        dtype=DEFAULT_DTYPE,
        seed=DEFAULT_SEED,
        **save_kwargs,  # Add prop kwargs if available
    )
    print(f"Coefficients saved to: {output_file}")
    if prescribed_R != "unknown":
        print(f"  Prescriber from model metadata: {prescribed_R}")
        if save_kwargs:
            print(f"  Original prop parameters: {save_kwargs}")
        print(f"  Prescriber (from model metadata): {prescribed_R}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
