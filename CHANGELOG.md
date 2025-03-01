# Change Log

## Unreleased

PyLOO has evolved into a comprehensive toolkit for model comparison and validation in Bayesian statistics. Our latest developments focus on both core functionality and advanced features for specific modeling frameworks.

### Core Functionality

We've built a flexible foundation for model assessment that works with any probabilistic programming language through ArviZ's InferenceData format. The core modules provide essential tools for model comparison:

The `loo` module implements leave-one-out cross-validation with support for multiple importance sampling methods (PSIS, SIS, TIS), comprehensive diagnostics, and flexible output scales. This allows users to reliably assess predictive performance without the computational burden of refitting models.

Our `waic` implementation provides the widely applicable information criterion as an alternative approach to model assessment, with the same flexible interface and output formats as LOO-CV.

For large datasets, we've developed efficient approximation methods in `loo_subsample`. This module uses statistical subsampling techniques to dramatically reduce computation time while maintaining accuracy. It supports multiple approximation methods (PLPD, LPD, TIS, SIS) and various estimator approaches (diff_srs, hh_pps, srs) with enhanced support for multidimensional data.

These core modules are complemented by a suite of supporting utilities, including pointwise LOO-CV calculations, expected LOO computations, and various importance sampling methods.

### Advanced PyMC Integration

We've created a universal PyMC wrapper, `pymc_wrapper`, that enables advanced cross-validation techniques specifically for PyMC models. This wrapper provides a standardized interface to model components while handling the complexities of PyMC's internal structures.

The `reloo` module implements exact refitting for problematic observations in LOO-CV. When importance sampling fails to provide reliable estimates for certain data points, this module automatically refits the model without those observations to get exact leave-one-out values.

Our `loo_kfold` module provides comprehensive K-fold cross-validation for PyMC models. It supports customizable fold creation, stratified sampling, and detailed diagnostics. This is particularly valuable for models where standard LOO approximations may be unreliable.

The PyMC wrapper handles all the complex details of working with PyMC models, including parameter transformations, data manipulation, and posterior sampling. This makes it straightforward to apply advanced cross-validation techniques to any PyMC model.

### Maintenance and Dependencies

Our core dependencies include:
- numpy>=1.20.0
- scipy>=1.7.0
- pandas>=1.3.0
- arviz>=0.15.0
- pymc>=5.0.0

### Documentation

We've expanded our documentation to include comprehensive usage examples for all major features:
- Standard LOO-CV with different importance sampling methods
- Subsampled LOO-CV with various approximation techniques
- Integration examples with ArviZ
- PyMC-specific workflows for reloo and k-fold cross-validation

The API documentation now covers all key components, including importance sampling methods, subsampling estimators, and diagnostic tools.

## v0.1.0 (2025-01-29)

### New features
- Project initialization and basic repository structure

### Maintenance and fixes
- Development tooling setup:
  - pytest for testing
  - black for code formatting
  - isort for import sorting
  - flake8 for linting
  - mypy for type checking
  - pre-commit hooks

[Unreleased]: https://github.com/your-username/pyloo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/pyloo/releases/tag/v0.1.0
