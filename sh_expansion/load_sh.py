"""  
Load and evaluate saved spherical harmonics coefficients.

This script:
1. Loads coefficients from a saved .npz file
2. Prints the coefficients
3. Generates test samples uniformly on the sphere
4. Evaluates and compares three versions of prescribed R:
   - Original prescribed R (from metadata)
   - R from SpectralPair built on learnt u coefficients
   - R from Laplace-Beltrami operator on learnt u
5. Creates 2D (theta, phi) plots of prescribed and SpectralPair R
6. Computes MSE between original and SpectralPair R

Usage:
    python sh_expansion/load_sh.py [run_number]
    
If run_number is not provided, loads the most recent run.
"""

import sys
from pathlib import Path

# Add parent directory to path BEFORE imports from sh_functions
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

# Configure which run to load (None = most recent)
RUN_NUMBER = None  # Set to specific number like 1, 2, 3, etc. or leave as None

# Configure plot color scale
fix_R_scales = False  # If True, use same color scale for all plots; if False, each plot has independent scale


class TeeOutput:
    """Context manager to duplicate output to both stdout and a file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.stdout = sys.stdout
        
    def __enter__(self):
        self.file = open(self.filepath, 'w')
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        if self.file:
            self.file.close()
            
    def write(self, data):
        self.stdout.write(data)
        if self.file:
            self.file.write(data)
            
    def flush(self):
        self.stdout.flush()
        if self.file:
            self.file.flush()

# Import shared functions
from sh_expansion.sh_functions import (
    evaluate_u,
    compute_laplace_beltrami_tf,
    compute_scalar_curvature,
    SphericalHarmonicsExpansion,
    xyz_to_spherical,
    EPSILON_COEFF_DISPLAY,
)
from data.prescribers import build_prescriber, SpectralPair
from data.samplers import StereoSampler
from network.global_conformal_model import GlobalConformalModel  # Import for model loading


def main():
    results_dir = Path(__file__).parent.parent / "results" / "sh_expansion"
    
    # Determine which run to load
    run_number = RUN_NUMBER
    if len(sys.argv) > 1:
        try:
            run_number = int(sys.argv[1])
        except ValueError:
            print(f"Error: Run number must be an integer, got '{sys.argv[1]}'")
            sys.exit(1)
    
    # If no run number specified, find the highest
    if run_number is None:
        if not results_dir.exists():
            print(f"Results directory does not exist: {results_dir}")
            sys.exit(1)
        
        coeff_files = list(results_dir.glob("coefficients_run*.npz"))
        if not coeff_files:
            print(f"No coefficient files found in {results_dir}")
            print("Run an optimization script first:")
            print("  - python sh_expansion/run_sh_direct.py --hps <config>  (direct method)")
            print("  - python sh_expansion/run_sh_model.py  (model-based method)")
            sys.exit(1)
        
        # Extract run numbers and find max
        run_numbers = []
        for f in coeff_files:
            try:
                num = int(f.stem.replace("coefficients_run", ""))
                run_numbers.append(num)
            except ValueError:
                pass
        
        if not run_numbers:
            print(f"No valid run files found in {results_dir}")
            sys.exit(1)
        
        run_number = max(run_numbers)
        print(f"Loading most recent run: {run_number}")
    
    # Create output directory for plots and log file
    output_dir = results_dir / f"run{run_number}_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start logging all output to file
    with TeeOutput(output_dir / 'load_output.txt'):
        # Load the specified run
        npz_path = results_dir / f"coefficients_run{run_number}.npz"
        if not npz_path.exists():
            print(f"Error: Run {run_number} not found at {npz_path}")
            print(f"\nAvailable runs:")
            coeff_files = sorted(results_dir.glob("coefficients_run*.npz"))
            if coeff_files:
                for f in coeff_files:
                    try:
                        num = int(f.stem.replace("coefficients_run", ""))
                        print(f"  - Run {num}: {f.name}")
                    except ValueError:
                        pass
            else:
                print("  None found")
            sys.exit(1)
        
        print(f"Loading coefficients from: {npz_path}")
        print("=" * 70)
        
        data = np.load(npz_path)
        coefficients = data['coefficients']
        lm_pairs = data['lm_pairs']
        loss = float(data['loss'])
        l_max = int(data['l_max']) if 'l_max' in data else int(data.get('max_degree', 6))
        
        # Load metadata (with fallbacks for older files)
        method = str(data.get('method', 'direct'))  # 'direct' or 'model_based'
        prescriber = str(data['prescriber']) if 'prescriber' in data else 'unknown'
        config_file = str(data['config_file']) if 'config_file' in data else None
        num_samples = int(data['num_samples']) if 'num_samples' in data else 'N/A'
        num_patches = int(data['num_patches']) if 'num_patches' in data else 'N/A'
        epochs = int(data['epochs']) if 'epochs' in data else 'N/A'
        learning_rate = float(data['learning_rate']) if 'learning_rate' in data else 'N/A'
        scheduler = str(data['scheduler']) if 'scheduler' in data else 'N/A'
        dtype = str(data['dtype']) if 'dtype' in data else 'N/A'
        model_checkpoint = str(data['model_checkpoint']) if 'model_checkpoint' in data else None
        
        print(f"Run: {run_number}")
        print(f"Method: {method}")
        if model_checkpoint:
            print(f"Model checkpoint: {model_checkpoint}")
        if method == 'direct':
            print(f"Prescriber: {prescriber}")
            if config_file:
                print(f"Config file: {config_file}")
        print(f"l_max: {l_max}")
        print(f"Number of coefficients: {len(coefficients)}")
        print(f"Best loss: {loss:.6e}")
        print()
        print("Training Configuration:")
        print(f"  Samples per patch: {num_samples}")
        print(f"  Number of patches: {num_patches}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        if method == 'direct':
            print(f"  Scheduler: {scheduler}")
        print(f"  Dtype: {dtype}")
        print()
        
        # Display significant coefficients (relative to largest coefficient)
        max_coeff = np.max(np.abs(coefficients))
        threshold = max_coeff * EPSILON_COEFF_DISPLAY
        print(f"Significant coefficients (|c| > {threshold:.6e}, i.e., {EPSILON_COEFF_DISPLAY*100:.1f}% of max):")
        print("-" * 70)
        for idx, (l, m) in enumerate(lm_pairs):
            coeff = coefficients[idx]
            if abs(coeff) > threshold:
                l_val = lm_pairs[idx][0]
                eff_coeff = coeff / (l_val * (l_val + 1))
                print(f"  c({l:2d}, {m:3d}) = {coeff:12.6e}  [effective in u: {eff_coeff:12.6e}]")
        print("-" * 70)
        print()
        
        # Generate uniform test samples on sphere using StereoSampler
        print("Generating test samples uniformly on sphere using StereoSampler...")
        n_test = 2000
        
        # Determine number of patches from metadata
        num_patches_metadata = int(data.get('num_patches', 2))
        
        # Create sampler (same as used in data/dataset.py)
        sampler = StereoSampler(
            num_patches=num_patches_metadata,
            num_samples=n_test,
            radial_offset=0.0,
            dtype=tf.float64,
        )
        
        # Generate samples - returns (patch_coords, xyz_coords)
        # patch_coords: (n_test, num_patches, 2) or (n_test, 2) if num_patches=1
        # xyz_coords: (n_test, num_patches, 3) or (n_test, 3) if num_patches=1
        patch_coords, xyz_coords = sampler()
        
        print(f"Generated {n_test} test samples per patch")
        print()
        
        # ========================================================================
        # 1. Original prescribed R (from metadata)
        # ========================================================================
        print("Loading original prescriber from metadata...")
    
        original_prescriber = None
        prescriber_kind = None
        prescriber_kwargs = {}
    
        # Extract prescriber information from metadata
        if 'prescriber' in data:
            prescriber_str = str(data['prescriber'])
            # Parse prescriber string (e.g., "prop(cs=1.0;ls=2;ms=0)")
            if '(' in prescriber_str:
                prescriber_kind = prescriber_str.split('(')[0]
                # For prop, try to load from saved arrays
                if prescriber_kind == 'prop':
                    if 'original_prop_ls' in data:
                        prescriber_kwargs['ls'] = data['original_prop_ls'].tolist()
                    if 'original_prop_cs' in data:
                        prescriber_kwargs['cs'] = data['original_prop_cs'].tolist()
                    if 'original_prop_ms' in data:
                        prescriber_kwargs['ms'] = data['original_prop_ms'].tolist()
            else:
                prescriber_kind = prescriber_str
    
        # If direct method and have config file, load from config
        if method == 'direct' and config_file and (prescriber_kind is None or not prescriber_kwargs):
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                if 'cfg' in config_data:
                    config_dict = config_data['cfg']
                else:
                    config_dict = config_data
            
                prescribed_R = config_dict.get('data', {}).get('prescribed_R', {})
                if isinstance(prescribed_R, dict):
                    prescriber_kind = prescribed_R.get('kind')
                    prescriber_kwargs = prescribed_R.get('kwargs', {})
    
        # For model-based method, load from model config
        if method == 'model_based' and model_checkpoint:
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / model_checkpoint
            model_path = checkpoint_dir / "final_model.keras"
            if model_path.exists():
                model = tf.keras.models.load_model(model_path, compile=False)
                if hasattr(model, 'cfg'):
                    prescribed_R_obj = model.cfg.data.prescribed_R
                    if isinstance(prescribed_R_obj, str):
                        prescriber_kind = prescribed_R_obj
                    else:
                        prescriber_kind = prescribed_R_obj.kind
                        if hasattr(prescribed_R_obj, 'kwargs'):
                            prescriber_kwargs = prescribed_R_obj.kwargs
    
        if prescriber_kind:
            print(f"Prescriber kind: {prescriber_kind}")
            if prescriber_kwargs:
                print(f"Prescriber kwargs: {prescriber_kwargs}")
            original_prescriber = build_prescriber(prescriber_kind, **prescriber_kwargs)
        
            # Evaluate original R on test samples using xyz coordinates
            # xyz_coords shape: (n_test, num_patches, 3) or (n_test, 3)
            if num_patches_metadata == 1:
                xyz_flat = xyz_coords  # (n_test, 3)
            else:
                # Need to match the ordering used in Laplace-Beltrami computation
                # Laplace concatenates patches: [patch0_all_samples, patch1_all_samples]
                # So we need: xyz_coords[:, 0, :] then xyz_coords[:, 1, :]
                xyz_flat = tf.concat([xyz_coords[:, i, :] for i in range(num_patches_metadata)], axis=0)
        
            R_original = original_prescriber(xyz_flat).numpy().flatten()
            print(f"Evaluated original R: mean={R_original.mean():.6f}, std={R_original.std():.6f}")
        else:
            print("Warning: Could not load original prescriber")
            R_original = None
        print()
    
        # ========================================================================
        # 2. R from SpectralPair built on learnt u coefficients
        # ========================================================================
        print("Building SpectralPair from learnt coefficients...")
    
        # Extract (l, m, c) from learnt coefficients
        learnt_ls = [int(lm[0]) for lm in lm_pairs]
        learnt_ms = [int(lm[1]) for lm in lm_pairs]
        learnt_cs = coefficients.tolist()
    
        spectral_prescriber = SpectralPair(ls=learnt_ls, cs=learnt_cs, ms=learnt_ms)
    
        # Use same xyz coordinates as original (matching patch ordering)
        if num_patches_metadata == 1:
            xyz_flat = xyz_coords  # (n_test, 3)
        else:
            # Match the ordering: [patch0_all_samples, patch1_all_samples]
            xyz_flat = tf.concat([xyz_coords[:, i, :] for i in range(num_patches_metadata)], axis=0)
    
        R_spectral = spectral_prescriber(xyz_flat).numpy().flatten()
        print(f"Evaluated SpectralPair R: mean={R_spectral.mean():.6f}, std={R_spectral.std():.6f}")
        print()
    
        # ========================================================================
        # 3. R from Laplace-Beltrami operator on learnt u
        # ========================================================================
        print("Computing R from Laplace-Beltrami operator...")
    
        # Create SphericalHarmonicsExpansion model
        u_model = SphericalHarmonicsExpansion(max_degree=l_max, dtype=tf.float64)
        u_model.coefficients.assign(tf.constant(coefficients, dtype=tf.float64))
    
        # Compute R for each patch using patch coordinates
        # patch_coords shape: (n_test, num_patches, 2) or (n_test, 2)
        R_laplace_list = []
    
        if num_patches_metadata == 1:
            # Single patch: patch_coords is (n_test, 2)
            coords = patch_coords  # (n_test, 2)
            u, delta_u = compute_laplace_beltrami_tf(u_model, coords, patch_idx=0)
            R_pred = compute_scalar_curvature(u, delta_u)
            R_laplace_list.append(R_pred)
        else:
            # Two patches: patch_coords is (n_test, 2, 2)
            for patch_idx in range(num_patches_metadata):
                coords = patch_coords[:, patch_idx, :]  # (n_test, 2)
                u, delta_u = compute_laplace_beltrami_tf(u_model, coords, patch_idx)
                R_pred = compute_scalar_curvature(u, delta_u)
                R_laplace_list.append(R_pred)
    
        # Concatenate all patches
        R_laplace = tf.concat(R_laplace_list, axis=0).numpy().flatten()
    
        print(f"Evaluated Laplace-Beltrami R: mean={R_laplace.mean():.6f}, std={R_laplace.std():.6f}")
        print()
    
        # Extract theta and phi from xyz coordinates for plotting
        if num_patches_metadata == 1:
            xyz_for_angles = xyz_coords  # (n_test, 3)
        else:
            # Match the ordering: [patch0_all_samples, patch1_all_samples]
            xyz_for_angles = tf.concat([xyz_coords[:, i, :] for i in range(num_patches_metadata)], axis=0)
    
        theta_test, phi_test = xyz_to_spherical(xyz_for_angles)
        theta_test = theta_test.numpy()
        phi_test = phi_test.numpy()
    
        # ========================================================================
        # Compute MSE losses
        # ========================================================================
        mse_spectral_vs_laplace = np.mean((R_spectral - R_laplace) ** 2)
        
        if R_original is not None:
            mse_spectral = np.mean((R_original - R_spectral) ** 2)
        
            print("=" * 70)
            print("MSE Losses:")
            print(f"  Original vs SpectralPair: {mse_spectral:.6e}")
            print(f"  SpectralPair vs Laplace-Beltrami: {mse_spectral_vs_laplace:.6e}")
            print("=" * 70)
        else:
            print("=" * 70)
            print("MSE Losses:")
            print(f"  SpectralPair vs Laplace-Beltrami: {mse_spectral_vs_laplace:.6e}")
            print("=" * 70)
        print()
    
        # ========================================================================
        # Create 2D plots
        # ========================================================================
        print("Creating plots...")
    
        # Determine common color scale if fix_R_scales is True
        if fix_R_scales:
            if R_original is not None:
                vmin_global = min(R_original.min(), R_spectral.min())
                vmax_global = max(R_original.max(), R_spectral.max())
            else:
                vmin_global = R_spectral.min()
                vmax_global = R_spectral.max()
        else:
            vmin_global = None
            vmax_global = None
    
        def create_plot(theta, phi, R_values, filename, vmin=None, vmax=None):
            """Create and save a 2D scatter plot of R values on (theta, phi)."""
            fig, ax = plt.subplots(figsize=(8, 6))
            # If vmin/vmax not provided (independent scales), use data range
            if vmin is None:
                vmin_plot = R_values.min()
                vmax_plot = R_values.max()
            else:
                vmin_plot = vmin
                vmax_plot = vmax
            scatter = ax.scatter(phi, theta, c=R_values, cmap='viridis', s=10, vmin=vmin_plot, vmax=vmax_plot)
            ax.set_xlabel('φ', fontsize=14)
            ax.set_ylabel('θ', fontsize=14)
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(0, np.pi)
            plt.colorbar(scatter, ax=ax, label='R')
            plt.tight_layout()
            plt.savefig(output_dir / filename, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {output_dir / filename}")
    
        if R_original is not None:
            create_plot(theta_test, phi_test, R_original, 'prescribed.pdf', vmin_global, vmax_global)
        create_plot(theta_test, phi_test, R_spectral, 'spectralpair.pdf', vmin_global, vmax_global)
    
        print()
        print("Done!")


if __name__ == "__main__":
    main()