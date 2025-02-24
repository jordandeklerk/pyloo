.. image:: ./assets/logo_symbol.png
   :width: 175
   :align: left

|active| |codecov| |black| |codacy| |commit_activity| |last_commit| |python|

.. |active| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: Project Status: Active

.. |codecov| image:: https://codecov.io/gh/jordandeklerk/pyloo/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/jordandeklerk/pyloo
   :alt: Code Coverage

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code Style

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/1c08ec7d782c451784293c996537de14
   :target: https://www.codacy.com/gh/jordandeklerk/pyloo/dashboard?utm_source=github.com&utm_medium=referral&utm_content=jordandeklerk/pyloo&utm_campaign=Badge_Grade
   :alt: Codacy Badge

.. |commit_activity| image:: https://img.shields.io/github/commit-activity/m/jordandeklerk/pyloo
   :target: https://github.com/jordandeklerk/pyloo/graphs/commit-activity
   :alt: Commit Activity

.. |last_commit| image:: https://img.shields.io/github/last-commit/jordandeklerk/pyloo
   :target: https://github.com/jordandeklerk/pyloo/graphs/commit-activity
   :alt: Last Commit

.. |python| image:: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue
   :target: https://www.python.org/
   :alt: Python version

**pyloo** is a framework-agnostic Python package providing efficient approximate leave-one-out cross-validation (LOO-CV) for fitted Bayesian models. This package has an R twin `loo <https://github.com/stan-dev/loo>`_.

.. warning::
   This project is in active development and not all features from the R package have been implemented yet. While the core functionality is available, some advanced features are still being worked on.

From existing posterior simulation draws, we compute approximate LOO-CV using Pareto smoothed importance sampling (PSIS), a procedure for regularizing importance weights. As a byproduct of our calculations, we also obtain approximate standard errors for estimated predictive errors, enabling robust model comparison and evaluation across multiple models.

The package implements the fast and stable computations for approximate LOO-CV from:

* Vehtari, A., Gelman, A., and Gabry, J. (2024). `Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC <https://arxiv.org/abs/1507.02646>`_. *Statistics and Computing*. 27(5), 1413--1432. doi:10.1007/s11222-016-9696-4.

Usage
-----

Standard PSIS-LOO-CV
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

.. code-block:: text

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

Model Comparison
~~~~~~~~~~~~~~

.. code-block:: python

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

.. code-block:: text

   Model comparison using LOO (scale: log)

                   elpd_loo   se    p_loo   weight    elpd_diff    dse
   non_centered     -11.2    2.1    3.1     0.62        0.0       0.0
   centered         -11.5    2.3    3.3     0.38       -0.3       0.4

   All Pareto k estimates are good (k < 0.7)

Advanced Usage
~~~~~~~~~~~~

For observations where PSIS-LOO approximation fails (indicated by large Pareto k values), pyloo can perform exact LOO-CV by refitting the model without those observations for PyMC models:

.. code-block:: python

   import pyloo as pl
   from pyloo.wrapper.pymc_wrapper import PyMCWrapper
   import pymc as pm
   import numpy as np

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

Installation
-----------

.. code-block:: bash

   pip install pyloo

Or with conda:

.. code-block:: bash

   conda install -c conda-forge pyloo

Resources
--------

* `Documentation <https://pyloo.readthedocs.io/>`_ (API reference, examples)
* `R package documentation <https://mc-stan.org/loo/reference/index.html>`_ (Additional methodology details)
* `Open an issue <https://github.com/jordandeklerk/pyloo/issues>`_ (Bug reports, feature requests)

Citation
--------

If you use pyloo in your research, please cite:

.. code-block:: bibtex

   @software{pyloo2025,
     author = {Jordan Deklerk},
     title = {pyloo: Python Implementation of LOO-CV and PSIS},
     year = {2025},
     publisher = {GitHub},
     url = {https://github.com/jordandeklerk/pyloo}
   }

For the underlying methodology, please also cite:

.. code-block:: bibtex

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
