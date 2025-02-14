<!-- <h1 align="center">
<img src="./assets/pyloo_logo.png"  width="300">
</h1> -->

<img src="./assets/logo_symbol.png"  width="150" align="left">

<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/pyloo.svg?label=Pypi%20downloads)](https://pypi.org/project/pyloo/) -->
<!-- [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyloo.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/pyloo) -->
[![Code Coverage](https://codecov.io/gh/jordandeklerk/pyloo/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/pyloo)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1c08ec7d782c451784293c996537de14)](https://www.codacy.com/gh/jordandeklerk/pyloo/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jordandeklerk/pyloo&amp;utm_campaign=Badge_Grade)
[![Python version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

__pyloo__ is a framework-agnostic Python implementation of the [R package loo](https://github.com/stan-dev/loo), providing efficient approximate leave-one-out cross-validation (LOO-CV) for fitted Bayesian models.
<br><br>
> ⚠️ **Note**: This project is in active development and not all features from the R package have been implemented yet. While the core functionality is available, some advanced features are still being worked on. We recommend checking the documentation for the current status of specific features.


From existing posterior simulation draws, we compute approximate LOO-CV using Pareto smoothed importance sampling (PSIS), a procedure for regularizing importance weights. As a byproduct of our calculations, we also obtain approximate standard errors for estimated predictive errors and for comparing predictive errors between two models.

The package implements the fast and stable computations for approximate LOO-CV from:

* Vehtari, A., Gelman, A., and Gabry, J. (2024). [Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC](https://arxiv.org/abs/1507.02646). _Statistics and Computing_. 27(5), 1413--1432. doi:10.1007/s11222-016-9696-4.

### Quick Usage

```python
import pyloo as pl
import arviz as az

data = az.load_arviz_data("centered_eight")

# Standard LOO-CV
loo_result = pl.loo(
    data,
    pointwise=True,  # Return pointwise values
    method="psis"    # Use PSIS (recommended)
)

print(loo_result)
```

For large datasets, we provide efficient subsampling-based computation:

```python
# LOO-CV with subsampling
subsample_result = pl.loo_subsample(
    data,
    observations=400,          # Subsample size
    loo_approximation="plpd",  # Use point estimate based approximation
    estimator="diff_srs"       # Use difference estimator
)

print(subsample_result)
```

### Installation

```bash
pip install pyloo
```

Or with conda:

```bash
conda install -c conda-forge pyloo
```

### Resources

* [Documentation](https://pyloo.readthedocs.io/) (API reference, examples)
* [R package documentation](https://mc-stan.org/loo/reference/index.html) (Additional methodology details)
* [Open an issue](https://github.com/jordandeklerk/pyloo/issues) (Bug reports, feature requests)

### Citation

If you use pyloo in your research, please cite:

```bibtex
@software{pyloo2025,
  author = {Jordan Deklerk},
  title = {pyloo: Python Implementation of LOO-CV and PSIS},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jordandeklerk/pyloo}
}
```

For the underlying methodology, please also cite:

```bibtex
@article{vehtari2024practical,
  title={Practical {B}ayesian model evaluation using leave-one-out cross-validation and {WAIC}},
  author={Vehtari, Aki and Gelman, Andrew and Gabry, Jonah},
  journal={Statistics and Computing},
  volume={27},
  number={5},
  pages={1413--1432},
  year={2024},
  publisher={Springer}
}
