.. raw:: html

   <img src="./assets/pyloo-dark.png#gh-light-mode-only" width="175" align="left" alt="pyloo logo" />
   <img src="./assets/pyloo-light.png#gh-dark-mode-only" width="175" align="left" alt="pyloo logo" />

|active| |build_status| |codecov| |black| |mypy| |commit_activity| |last_commit| |python|

.. |active| image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/#active
   :alt: Project Status: Active

.. |build_status| image:: https://github.com/jordandeklerk/pyloo/actions/workflows/test.yml/badge.svg
   :target: https://github.com/jordandeklerk/pyloo/actions/workflows/test.yml
   :alt: Build Status

.. |codecov| image:: https://codecov.io/gh/jordandeklerk/pyloo/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/jordandeklerk/pyloo
   :alt: Code Coverage

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code Style

.. |mypy| image:: https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python
   :target: https://mypy-lang.org/
   :alt: Type Checked

.. |last_commit| image:: https://img.shields.io/github/last-commit/jordandeklerk/pyloo
   :target: https://github.com/jordandeklerk/pyloo/graphs/commit-activity
   :alt: Last Commit

.. |commit_activity| image:: https://img.shields.io/github/commit-activity/m/jordandeklerk/pyloo
   :target: https://github.com/jordandeklerk/pyloo/graphs/commit-activity
   :alt: Commit Activity

.. |python| image:: https://img.shields.io/badge/3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white
   :target: https://www.python.org/
   :alt: Python version

__pyloo__ is a Python package providing efficient approximate leave-one-out cross-validation (LOO-CV) for fitted Bayesian models with advanced features for **PyMC** models. Inspired by its R twin `loo <https://github.com/stan-dev/loo>`_, __pyloo__ brings similar functionality to the Python ecosystem.
<br><br>

The package implements the fast and stable computations for approximate LOO-CV from:

* Vehtari, A., Gelman, A., and Gabry, J. (2017). `Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC <https://link.springer.com/article/10.1007/s11222-016-9696-4>`_. *Statistics and Computing*. 27(5), 1413--1432. doi:10.1007/s11222-016-9696-4. (`arXiv preprint <https://arxiv.org/abs/1507.04544>`_).

and computes model weights as described in:

* Yao, Y., Vehtari, A., Simpson, D., and Gelman, A. (2018). `Using stacking to average Bayesian predictive distributions <https://projecteuclid.org/euclid.ba/1516093227>`_. *Bayesian Analysis*. doi:10.1214/17-BA1091. (`arXiv preprint <https://arxiv.org/abs/1704.02030>`_).

We recommend PSIS-LOO-CV over WAIC because PSIS offers informative diagnostics (like Pareto k estimates) and estimates for effective sample size and Monte Carlo error, providing greater insight into the reliability of the results.

Features
--------

Core Functionality
~~~~~~~~~~~~~~~~~~

*   **Approximate Leave-One-Out Cross-Validation (LOO-CV)**: Efficiently compute LOO-CV using Pareto Smoothed Importance Sampling (PSIS) and other methods, complete with diagnostics (``loo``).
*   **Model Comparison**: Compare models based on ELPD using various methods like stacking or Bayesian bootstrap (``loo_compare``).
*   **LOO-Based Metrics**: Estimate predictive performance (e.g., MAE, MSE, CRPS) using LOO estimates (``loo_predictive_metric``, ``loo_score``).
*   **Non-factorized LOO-CV**: Compute LOO-CV for multivariate normal and Student-t models where the likelihood cannot be factorized by observations (``loo_nonfactor``).
*   **Grouped & Subsampled LOO-CV**: Perform Leave-One-Group-Out CV (``loo_group``) and efficient subsampling for large datasets (``loo_subsample``).
*   **Widely Applicable Information Criterion (WAIC)**: Calculate WAIC as an alternative model assessment metric (``waic``).

Advanced & PyMC Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **PyMC Model Interface**: Seamless integration with PyMC models for streamlined workflow (``PyMCWrapper``).
*   **Moment Matching**: Improve LOO-CV reliability for challenging observations by matching moments (``loo_moment_match``).
*   **Exact Refitting (Reloo)**: Refit models for problematic observations identified by LOO diagnostics (``reloo``).
*   **K-Fold Cross-Validation**: Flexible K-fold CV implementation with stratification, groups, and diagnostics (``loo_kfold``).
*   **Variational Inference Support**: Compute LOO-CV for models fitted with Laplace or ADVI variational approximations (``loo_approximate_posterior``).

Quickstart
----------

PSIS-LOO-CV
~~~~~~~~~~~

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

   Computed from 2000 posterior samples and 8 observations log-likelihood matrix.

          Estimate       SE
   elpd_loo   -30.78      1.35
   p_loo       0.95        0.48
   looic      61.56       2.69

   All Pareto k estimates are good (k < 0.7).
   See help('pareto-k-diagnostic') for details.

Model Comparison
~~~~~~~~~~~~~~

Compare multiple models with `compare` using stacking weights or other methods:

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

                 elpd_loo   se      p_loo   weight    elpd_diff    dse     warning
   non_centered     -30.72   1.33     0.90    1.00e+00     0.0      0.00      True
   centered         -30.78   1.35     0.95    2.50e-16    -0.06     0.06      False

   All Pareto k estimates are good (k < 0.7)

Installation
------------

.. warning::
   **Note**: ``pyloo`` is currently under active development and has not yet been officially released or published to PyPI. While the core functionality is being stabilized, expect potential changes to the API and features. Installation is currently only possible directly from the development version on GitHub (see below).

.. code-block:: bash

   pip install pyloo

Or with conda:

.. code-block:: bash

   conda install -c conda-forge pyloo

Development Version
^^^^^^^^^^^^^^^^^

To install the latest development version directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/jordandeklerk/pyloo.git

For development purposes, you can clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/jordandeklerk/pyloo.git
   cd pyloo
   pip install -e .

Resources
---------

.. warning::
   **Note**: Documentation coming soon.

* `Documentation <>`_ (API reference, examples)
* `Open an issue <>`_ (Bug reports, feature requests)

Citation
--------

.. code-block:: bibtex

   @article{vehtari2024practical,
     title={Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC},
     author={Vehtari, Aki and Gelman, Andrew and Gabry, Jonah},
     journal={Statistics and Computing},
     volume={27},
     number={5},
     pages={1413--1432},
     year={2017},
     publisher={Springer}
   }

   @article{yao2018using,
     title={Using stacking to average Bayesian predictive distributions},
     author={Yao, Yuling and Vehtari, Aki and Simpson, Daniel and Gelman, Andrew},
     journal={Bayesian Analysis},
     volume={13},
     number={3},
     pages={911--1007},
     year={2018},
     publisher={International Society for Bayesian Analysis}
   }
