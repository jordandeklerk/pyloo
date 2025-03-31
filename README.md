<!-- <h1 align="center">
<img src="./assets/pyloo_logo_revised.png"  width="250">
</h1> -->

<img src="./assets/pyloo_logo.png"  width="200" align="left">

<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/pyloo.svg?label=Pypi%20downloads)](https://pypi.org/project/pyloo/) -->
<!-- [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyloo.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/pyloo) -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/pyloo/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/pyloo)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1c08ec7d782c451784293c996537de14)](https://www.codacy.com/gh/jordandeklerk/pyloo/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jordandeklerk/pyloo&amp;utm_campaign=Badge_Grade)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/pyloo)](https://github.com/jordandeklerk/pyloo/graphs/commit-activity)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/pyloo)](https://github.com/jordandeklerk/pyloo/graphs/commit-activity)
[![Python version](https://img.shields.io/badge/3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

__PyLOO__ is a framework-agnostic Python package providing efficient approximate leave-one-out cross-validation (LOO-CV) for Bayesian models with advanced features for **PyMC** models. Inspired by its R twin [loo](https://github.com/stan-dev/loo), PyLOO brings similar functionality to the Python ecosystem.
<br><br>
> ⚠️ **Note**: This project is in active development and not all features from the R package have been implemented yet. While the core functionality is available, some advanced features are still being developed.

The package implements the fast and stable computations for approximate LOO-CV from:

* Vehtari, A., Gelman, A., and Gabry, J. (2024). [Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC](https://arxiv.org/abs/1507.02646). _Statistics and Computing_. 27(5), 1413--1432. doi:10.1007/s11222-016-9696-4.

## Features

### Core

- **LOO-CV**: Leave-one-out cross-validation with multiple importance sampling methods (PSIS, SIS, TIS), comprehensive diagnostics, and flexible output scales.
- **WAIC**: Widely Applicable Information Criterion as an alternative approach to model assessment, with consistent interface and output formats.
- **Efficient Subsampling**: Statistical subsampling techniques for large datasets that reduce computation time while maintaining accuracy.
- **Model Comparison**: Compare models based on their expected log pointwise predictive density (ELPD).

### Advanced
- **Universal PyMC Wrapper**: Standardized interface to model components that manages parameter transformations, data manipulation, posterior sampling, and pointwise log-likelihood computations.
- **Reloo**: Exact refitting for problematic observations in LOO-CV when importance sampling fails to provide reliable estimates.
- **K-fold Cross-validation**: Comprehensive K-fold CV with customizable fold creation, stratified sampling, and detailed diagnostics.
- **Moment Matching**: Transforms posterior draws to better approximate leave-one-out posteriors, improving reliability of LOO-CV estimates for observations with high Pareto k diagnostics.
- **Posterior Approximations**: Compute LOO-CV for posterior approximations supporting Laplace, ADVI and Full-Rank ADVI variational approximations.

## Quickstart

### PSIS-LOO-CV

```python
import pyloo as pl
import arviz as az

data = az.load_arviz_data("centered_eight")

# PSIS-LOO-CV
loo_result = pl.loo(
    data,
    pointwise=True,  # Return pointwise values
    method="psis"    # Use PSIS (recommended)
)

print(loo_result)
```
```
Computed from 2000 posterior samples and 8 observations log-likelihood matrix.

         Estimate       SE
elpd_loo   -30.78      1.35
p_loo       0.95        0.48
looic      61.56       2.69

All Pareto k estimates are good (k < 0.7).
See help('pareto-k-diagnostic') for details.
```

### Model Comparison

Compare multiple models with `compare` using stacking weights or other methods:

```python
model1 = az.load_arviz_data("centered_eight")
model2 = az.load_arviz_data("non_centered_eight")

comparison = pl.loo_compare(
    {
        "centered": model1,
        "non_centered": model2
    },
    ic="loo",                # Information criterion to use
    method="stacking",       # Method for computing weights
    scale="log"              # Scale for the scores
)

print(comparison)
```
```
Model comparison using LOO (scale: log)

                elpd_loo   se      p_loo   weight    elpd_diff    dse     warning
non_centered     -30.72   1.33     0.90    1.00e+00     0.0      0.00      True
centered         -30.78   1.35     0.95    2.50e-16    -0.06     0.06      False

All Pareto k estimates are good (k < 0.7)
```

## Installation

> ⚠️ **Note**: Not yet available for installation from PyPI.

```bash
pip install pyloo
```

Or with conda:

```bash
conda install -c conda-forge pyloo
```

### Development Version

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/jordandeklerk/pyloo.git
```

For development purposes, you can clone the repository and install in editable mode:

```bash
git clone https://github.com/jordandeklerk/pyloo.git
cd pyloo
pip install -e .
```

## Resources

> ⚠️ **Note**: Documentation coming soon.

* [Documentation]() (API reference, examples)
* [Open an issue]() (Bug reports, feature requests)

## Citation

```bibtex
@article{vehtari2024practical,
  title={Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC},
  author={Vehtari, Aki and Gelman, Andrew and Gabry, Jonah},
  journal={Statistics and Computing},
  volume={27},
  number={5},
  pages={1413--1432},
  year={2024},
  publisher={Springer}
}
