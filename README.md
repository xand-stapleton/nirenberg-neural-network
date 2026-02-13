# Nirenberg Neural Network (NNN)

A neural network framework for approaching the Nirenberg problem. A Physics-Informed Neural Network (PINN) architecture learns the conformal factor, $u$, that defines a metric $g=e^{2u}g_0$ with a desired prescribed scalar curvature.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For detailed setup and contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start

New to the project? Try **[quick_start.ipynb](quick_start.ipynb)** for an interactive walkthrough that demonstrates loading a configuration, running training, and exploring results.

## Running Training

Training is executed via [run.py](run.py) with configuration files that define all hyperparameters and prescribers:

```bash
python run.py --hps configs/<config_file>.yaml
```

Configuration files are organized in [configs/](configs/):
- `configs/known/` - Known analytic solutions
- `configs/unknown/` - Unknown target metrics
- `configs/sh/` - Spherical harmonic basis configurations

## Outputs

- **Trained models**: Saved to [checkpoints/](checkpoints/) as `.keras` files
- **Results**: Training metrics, plots, and analysis data saved to [results/](results/)

## Analysis Notebooks

Two Jupyter notebooks are provided for analyzing trained models:

- **[examine_output.ipynb](examine_output.ipynb)** - Primary analysis tool for examining model predictions, comparing against ground truth, visualizing metrics, and evaluating solution quality
- **[metric_visualisation.ipynb](metric_visualisation.ipynb)** - Generates MDS embedding visualisations of trained models

## Spherical Harmonic Expansion

The [sh_expansion/](sh_expansion/) directory contains tools for fitting spherical harmonic expansions to trained models. See [sh_expansion/README.md](sh_expansion/README.md) for details

## Citation

If you use this code in your research, please cite:

```bibtex
    raise NotImplementedError("Paper yet to be posted.")
```
