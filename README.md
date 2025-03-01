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

__pyloo__ is a framework-agnostic Python package providing efficient approximate leave-one-out cross-validation (LOO-CV) for fitted Bayesian models. This package has an R twin [loo](https://github.com/stan-dev/loo).
<br><br><br>
> ⚠️ **Note**: This project is in active development and not all features from the R package have been implemented yet. While the core functionality is available, some advanced features are still being developed.


From existing posterior simulation draws, we compute approximate LOO-CV using Pareto smoothed importance sampling (PSIS), a procedure for regularizing importance weights. As a byproduct of our calculations, we also obtain approximate standard errors for estimated predictive errors, enabling robust model comparison and evaluation across multiple models.


The package implements the fast and stable computations for approximate LOO-CV from:

* Vehtari, A., Gelman, A., and Gabry, J. (2024). [Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC](https://arxiv.org/abs/1507.02646). _Statistics and Computing_. 27(5), 1413--1432. doi:10.1007/s11222-016-9696-4.

### Usage

#### Standard PSIS-LOO-CV
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
Computed from 4000 samples using all 8 observations.

           Estimate   SE
elpd_loo   -11.2     2.1
p_loo       3.1      -
looic       22.4     4.2

All Pareto k estimates are good (k < 0.7).
See help('pareto-k-diagnostic') for details.

Pareto k diagnostic values:
                          Count    Pct.
(-Inf, 0.70)                 8   100.0
[0.70, 1)                    0     0.0
[1, Inf)                     0     0.0
```

#### Model Comparison
Compare multiple models using stacking weights or other methods:

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

                elpd_loo   se    p_loo   weight    elpd_diff    dse
non_centered     -11.2    2.1    3.1     0.62        0.0       0.0
centered         -11.5    2.3    3.3     0.38       -0.3       0.4

All Pareto k estimates are good (k < 0.7)
```

### Advanced Usage
We provide several advanced features beyond the core capabilities for fitted PyMC models.

#### Reloo
For observations where PSIS-LOO approximation fails (indicated by large Pareto k values), pyloo can perform exact LOO-CV by refitting the model without those observations:

```python
import pyloo as pl
import pymc as pm
import numpy as np

from pyloo.wrapper.pymc_wrapper import PyMCWrapper

np.random.seed(0)
N = 100
y = np.random.normal(1.0, 2.0, N)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    idata = pm.sample(1000, tune=1000)

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

#### K-fold Cross-Validation
When you have a moderate amount of data or when individual observations have strong influence on the model, K-fold cross-validation can provide a more stable estimate of out-of-sample predictive performance than LOO-CV:

```python
import pyloo as pl
import pymc as pm
import numpy as np

from pyloo.wrapper.pymc_wrapper import PyMCWrapper

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
kfold_result = pl.kfold(wrapper, K=5)
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
kfold_result = pl.kfold(
    wrapper,
    K=5,
    stratify=wrapper.get_observed_data(),  # Stratify by outcome
    random_seed=123
)
```

You can also save the fitted models for each fold for further analysis:

```python
kfold_with_fits = pl.kfold(
    wrapper,
    K=5,
    save_fits=True,  # Save the fitted models
    random_seed=123
)

# Access the fits for the first fold
first_fold_idata, first_fold_indices = kfold_with_fits.fits[0]
```

### Installation

> ⚠️ **Note**: Not yet available for installation from PyPI.

```bash
pip install pyloo
```

Or with conda:

```bash
conda install -c conda-forge pyloo
```

#### Development Version

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

### Resources

> ⚠️ **Note**: Documentation coming soon.

* [Documentation]() (API reference, examples)
* [R package documentation]() (Additional methodology details)
* [Open an issue]() (Bug reports, feature requests)

### Citation

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
