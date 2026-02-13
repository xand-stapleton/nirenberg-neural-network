# Spherical Harmonics Expansion

Alternative methods for solving the Nirenberg problem using optimization of spherical harmonics coefficients.

The conformal factor u is parameterized as a finite spherical harmonics expansion:

$$u(\theta, \phi) = \sum_{l=1}^{l_{\text{max}}} \sum_{m=-l}^{l} \frac{c_{l,m}}{l(l+1)} \, Y_l^m(\theta, \phi)$$

where $Y_l^m$ are real-valued spherical harmonics and $c_{l,m}$ are learnable coefficients. Note: the sum starts at $l=1$ (not $l=0$) to avoid division by zero in the conformal factor formula. Default: $l_{\text{max}} = 4$ (24 coefficients).

## Files

- **`sh_functions.py`**: Shared utility functions and constants for spherical harmonics computations
- **`run_sh_direct.py`**: Direct optimization method - fits SH coefficients by computing scalar curvature from first principles
- **`run_sh_model.py`**: Model-based method - fits SH coefficients to match predictions from a trained neural network
- **`load_sh.py`**: Load and evaluate saved coefficients from either method

## Setup

Activate the virtual environment:

```bash
# From project root directory
source .venv/bin/activate
```

## Running the Scripts

### 1. Direct Optimization Method

Train a spherical harmonics expansion model by directly computing scalar curvature:

```bash
# From project root
python sh_expansion/run_sh_direct.py --config configs/known/prop_a.yaml
```

**Parameters:**
- `--config`: Path to config file (used only for prescriber configuration)
- `--l_max`: Maximum degree of spherical harmonics expansion (default: 4)
- `--lr`: Learning rate for Adam optimizer (default: 0.01)
- `--num_samples`: Number of samples to generate per patch (default: 5000)
- `--epochs`: Number of optimization epochs (default: 300)

**Note:** All parameters except the prescriber are specified via command-line arguments, not from the config file.

**Output:**
- Coefficients saved to: `results/sh_expansion/coefficients_run{N}.npz` (auto-numbered)
- Training logs printed to console

### 2. Model-Based Method

Extract spherical harmonics representation from a trained neural network model:

```bash
# From project root
python sh_expansion/run_sh_model.py --model watery-pond-1234
```

**Parameters:**
- `--model`: Model checkpoint folder name in `checkpoints/` (default: "watery-pond-1234")
- `--l_max`: Maximum degree of SH expansion (default: 4)
- `--num_samples`: Number of sphere samples for fitting (default: 5000)
- `--epochs`: Optimization epochs (default: 300)
- `--lr`: Learning rate (default: 0.01)

**Note:** Uses hardcoded defaults for `num_patches` (2), `dtype` (float64), and `seed` (42) defined in `sh_functions.py`.

**Output:**
- Coefficients saved to: `results/sh_expansion/coefficients_run{N}.npz` (auto-numbered)
- Includes model checkpoint info in metadata
- WandB logging (if enabled)

**Use case:** This method is useful when you already have a trained neural network solution and want to extract an analytical spherical harmonics approximation for easier analysis or visualization.

### 3. Load and Evaluate Coefficients

Load saved coefficients and compare three methods of computing scalar curvature:

```bash
# From project root - loads most recent run
python sh_expansion/load_sh.py

# Load specific run number
python sh_expansion/load_sh.py 1
```

**Configuration:**
Set `RUN_NUMBER` at the top of `load_sh.py` to specify a default run, or pass as command-line argument.

**What it does:**
1. Prints significant coefficients with effective values in u
2. Generates uniform test samples on sphere using `StereoSampler`
3. Computes and compares three versions of prescribed R:
   - Original R from prescriber metadata
   - R from `SpectralPair` built on learnt u coefficients (analytical)
   - R from Laplace-Beltrami operator on learnt u (numerical PDE)
4. Creates 2D (θ, φ) plots of each R with shared color scale
5. Reports MSE between methods (validates numerical vs analytical agreement)

**Output:**
- Console: Coefficients, statistics, and MSE losses
- Plots: `results/sh_expansion/run{N}_plots/{prescribed,spectralpair,laplace}.png`
- TextFile: The printed logs

### Saved Files

**Location:** `results/sh_expansion/`

**Format:** `.npz` file containing:
- `coefficients`: NumPy array of spherical harmonic coefficients
- `lm_pairs`: Corresponding (l, m) indices
- `loss`: Best loss achieved
- `l_max`: Maximum degree used
- `method`: "direct" or "model_based"
- Run metadata:
  - **Direct method**: prescriber, samples, epochs, learning rate, scheduler, dtype
  - **Model-based method**: model_checkpoint, samples, epochs, learning rate, dtype
