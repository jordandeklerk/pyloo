<img src="https://raw.githubusercontent.com/jordandeklerk/pyloo/main/assets/pyloo-logo.png" width="200" height="200" align="left">

[![PyPI Downloads](https://img.shields.io/pypi/dm/pyloo.svg?label=Pypi%20downloads)](https://pypi.org/project/pyloo/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyloo.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/pyloo)
[![Code Coverage](https://codecov.io/gh/jordandeklerk/pyloo/branch/main/graph/badge.svg)](https://codecov.io/gh/jordandeklerk/pyloo)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

pyloo is an open source Python implementation of the R package [loo](https://github.com/stan-dev/loo).
This package provides tools for Leave-One-Out Cross-Validation (LOO-CV)
and Pareto Smoothed Importance Sampling (PSIS) for Bayesian models in a framework agnostic way.

<br>

Please note that this project is in its early stages and is in active development. Our goal is to make reliable and robust model comparison accessible and straightforward for the Python-based Bayesian modeling community.

## Roadmap

Our vision is to provide a comprehensive, framework-agnostic workflow for Leave-One-Out Cross-Validation in Bayesian modeling. Here's what we're working on:

### Core PSIS-LOO-CV Implementation
- **Framework Agnosticism**: Support for any Bayesian modeling framework (PyMC, NumPyro, Stan, etc.) through a unified interface
- **Efficient LOO-CV**: Implementation of PSIS-LOO-CV for computationally efficient model assessment
- **Diagnostic Tools**: Comprehensive diagnostics including Pareto k values, effective sample size, Monte Carlo standard errors, and more
- **Model Comparison**: Modules implementing entire workflows for comparing multiple Bayesian models using LOO-CV techniques

### Advanced Features
- **Exact Refitting**: Implementation of exact model refitting for observations with problematic Pareto k values
- **Moment Matching**: Support for moment matching techniques to avoid the computational cost of exact refitting
- **Mixture Importance Sampling**: Implementation of mixture importance sampling as an alternative approach to handle challenging cases in high-dimensional settings

### Extended Functionality
- **K-Fold Cross-Validation**: Support for K-fold CV when LOO-CV is not suitable
- **Pointwise Diagnostics**: Detailed diagnostics for identifying influential observations
- **Visualization Tools**: Built-in plotting functions for pragmatic diagnostics and model comparison
