<!-- <h1 align="center">
<img src="./assets/pyloo_logo.png"  width="300">
</h1> -->

<img src="./assets/logo_symbol.png"  width="175" align="left">

<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/pyloo.svg?label=Pypi%20downloads)](https://pypi.org/project/pyloo/) -->
<!-- [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyloo.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/pyloo) -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/pyloo/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/pyloo)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1c08ec7d782c451784293c996537de14)](https://www.codacy.com/gh/jordandeklerk/pyloo/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jordandeklerk/pyloo&amp;utm_campaign=Badge_Grade)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/pyloo)](https://github.com/jordandeklerk/pyloo/graphs/commit-activity)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/pyloo)](https://github.com/jordandeklerk/pyloo/graphs/commit-activity)
[![Python version](https://img.shields.io/badge/3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

__pyloo__ is a framework-agnostic Python package providing efficient approximate leave-one-out cross-validation (LOO-CV) for Bayesian models with advanced features for **PyMC** models. Inspired by its R twin [loo](https://github.com/stan-dev/loo), pyloo brings similar functionality to the Python ecosystem.
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

### Reloo
For observations where PSIS-LOO approximation fails, we can perform exact LOO-CV with `reloo`:

```python
import pyloo as pl
import pymc as pm
import numpy as np

from pyloo.wrapper.pymc import PyMCWrapper

np.random.seed(0)
N = 100
y = np.random.normal(1.0, 2.0, N)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    idata = pm.sample(1000, chains=4, return_inferencedata=True,
                     idata_kwargs={"log_likelihood": True})

# Wrap the model in the PyMC wrapper
wrapper = PyMCWrapper(model, idata)
loo_exact = pl.reloo(wrapper, k_thresh=0.7)

# For large datasets, use subsampling
loo_exact_subsample = pl.reloo(
    wrapper,
    k_thresh=0.7,
    use_subsample=True,
    subsample_observations=50  # Use 50 observations
)
```

### K-fold Cross-Validation
When you have a moderate amount of data or when individual observations have strong influence on the model, K-fold cross-validation with `loo_kfold` can provide a more stable estimate of out-of-sample predictive performance than LOO-CV:

```python
import pyloo as pl
import pymc as pm
import numpy as np

from pyloo.wrapper.pymc import PyMCWrapper

np.random.seed(42)
x = np.random.normal(0, 1, size=100)
true_alpha = 1.0
true_beta = 2.5
true_sigma = 1.0
y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, size=100)

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)

    mu = alpha + beta * x
    obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(1000, chains=4, return_inferencedata=True,
                     idata_kwargs={"log_likelihood": True})

# Wrap the model in the PyMC wrapper
wrapper = PyMCWrapper(model, idata)

# Perform 5-fold cross-validation
kfold_result = pl.loo_kfold(wrapper, K=5)
```

For datasets with imbalanced features or outcomes, stratified K-fold cross-validation can provide more reliable performance estimates:

```python
np.random.seed(42)
n_samples = 200
y = np.random.binomial(1, 0.3, size=n_samples)

x1 = np.random.normal(y, 1.0)  # Correlated with outcome
x2 = np.random.normal(0, 1.0, size=n_samples)  # Independent feature
X = np.column_stack((x1, x2))

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=2)
    beta = pm.Normal("beta", mu=0, sigma=2, shape=2)

    logit_p = alpha + pm.math.dot(X, beta)
    obs = pm.Bernoulli("y", logit_p=logit_p, observed=y)

    idata = pm.sample(1000, chains=2, return_inferencedata=True,
                     idata_kwargs={"log_likelihood": True})

wrapper = PyMCWrapper(model, idata)

# Use stratified k-fold CV to maintain class distribution across folds
kfold_result = pl.loo_kfold(
    wrapper,
    K=5,
    stratify=wrapper.get_observed_data(),  # Stratify by outcome
    random_seed=123
)
```

### Moment Matching

When PSIS-LOO approximation fails, moment matching with `loo_moment_match` can improve the reliability of LOO-CV estimates without the computational cost of refitting the model:

```python
import pyloo as pl
import pymc as pm
import numpy as np
import arviz as az

from pyloo.wrapper.pymc import PyMCWrapper

np.random.seed(123)
N = 50
x = np.random.normal(0, 1, N)
y = 2 + 3 * x + np.random.normal(0, 1, N)
# Add an outlier
y[0] = 10

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + beta * x
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    idata = pm.sample(1000, chains=4, return_inferencedata=True,
                     idata_kwargs={"log_likelihood": True})

wrapper = PyMCWrapper(model, idata)
loo_orig = pl.loo(idata, pointwise=True)
```

After computing standard LOO-CV, you can apply moment matching to improve estimates for observations with high Pareto k values:

```python
loo_improved = pl.loo_moment_match(
    wrapper,
    loo_orig,
    max_iters=30,
    k_threshold=0.7,
    split=True,       # Use split moment matching for better stability
    cov=True,         # Match covariance matrix in addition to means and variances
    method="psis"
)
```

Alternatively, you can compute LOO-CV with moment matching by setting `moment_match = True` in the main `loo` function. This method also requires passing the PyMC wrapper to `loo`:

```python
loo_direct = pl.loo(
    idata,
    pointwise=True,
    moment_match=True,
    wrapper=wrapper,    # Required for moment matching
    k_threshold=0.7,
    split=True,
    cov=True,
    method="psis"
)
```

### Posterior Approximations

When working with posterior approximations, we can use `loo_approximate_posterior` to compute LOO-CV:

```python
import pyloo as pl
import pymc as pm
import numpy as np
import arviz as az

from pyloo.wrapper.laplace import Laplace

np.random.seed(42)
N = 100
x = np.random.normal(0, 1, N)
y = 2 + 3 * x + np.random.normal(0, 1, N)

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + beta * x
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

wrapper = Laplace(model)
laplace_result = wrapper.fit(
    optimize_method="BFGS",
    chains=4,
    draws=1000,
    compute_log_likelihood=True
)

log_p = wrapper.compute_logp()  # True posterior log density
log_q = wrapper.compute_logq()  # Approximation log density

loo_result = pl.loo_approximate_posterior(
    laplace_result.idata,
    log_p=log_p,
    log_q=log_q,
    pointwise=True,
    method="psis"
)
```

For large datasets, you can combine posterior approximations with sub-sampling for even more efficient computation:

```python
n_obs = laplace_result.idata.log_likelihood["y"].shape[2]
subsample_size = min(50, n_obs // 2)

loo_subsample_result = pl.loo_subsample(
    laplace_result.idata,
    observations=subsample_size,
    log_p=log_p,
    log_q=log_q,
    pointwise=True,
    loo_approximation="plpd",
    estimator="diff_srs"
)


updated_result = pl.update_subsample(
    loo_subsample_result,
    observations=min(75, n_obs // 2)
)
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
* [R package documentation]() (Additional methodology details)
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
