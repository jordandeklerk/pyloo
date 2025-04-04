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

.. |python| image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue
   :target: https://www.python.org/
   :alt: Python version

**PyLOO** is a framework-agnostic Python package providing efficient approximate leave-one-out cross-validation (LOO-CV) for Bayesian models with advanced features for **PyMC** models. Inspired by its R twin `loo <https://github.com/stan-dev/loo>`_, **PyLOO** brings similar functionality to the Python ecosystem.

.. warning::
   This project is in active development and not all features from the R package have been implemented yet. While the core functionality is available, some advanced features are still being developed.

The package implements the fast and stable computations for approximate LOO-CV from:

* Vehtari, A., Gelman, A., and Gabry, J. (2024). `Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC <https://arxiv.org/abs/1507.02646>`_. *Statistics and Computing*. 27(5), 1413--1432. doi:10.1007/s11222-016-9696-4.

Features
-------

PyLOO offers a comprehensive suite of tools for Bayesian model evaluation and comparison:

Core Functionality
~~~~~~~~~~~~~~~~

- **Approximate Leave-One-Out Cross-Validation (LOO-CV)**: Efficiently compute LOO-CV using Pareto Smoothed Importance Sampling (PSIS) and other methods, complete with diagnostics (``loo``).
- **Widely Applicable Information Criterion (WAIC)**: Calculate WAIC as an alternative model assessment metric (``waic``).
- **Model Comparison**: Compare models based on ELPD using various methods like stacking (``loo_compare``).
- **LOO-Based Metrics**: Estimate predictive performance (e.g., MAE, MSE, CRPS) using LOO estimates (``loo_predictive_metric``, ``loo_score``).
- **Grouped & Subsampled CV**: Perform Leave-One-Group-Out CV (``loo_group``) and efficient subsampling for large datasets (``loo_subsample``).

Advanced & PyMC Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

- **PyMC Model Interface**: Seamless integration with PyMC models for streamlined workflow.
- **Exact Refitting (Reloo)**: Refit models for problematic observations identified by LOO diagnostics (``reloo``).
- **K-Fold Cross-Validation**: Flexible K-fold CV implementation with stratification and diagnostics (``loo_kfold``).
- **Moment Matching**: Improve LOO-CV reliability for challenging observations (``loo_moment_match``).
- **Variational Inference Support**: Compute LOO-CV for models fitted with Laplace or ADVI approximations (``loo_approximate_posterior``).

Quickstart
---------

PSIS-LOO-CV
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
-----------

.. warning::
   Not yet available for installation from PyPI.

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
--------

.. warning::
   Documentation coming soon.

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
     year={2024},
     publisher={Springer}
   }
