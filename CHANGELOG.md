# Change Log

## Unreleased

PyLOO has evolved into a comprehensive toolkit for model comparison and validation in Bayesian statistics. Our latest developments focus on both core functionality and advanced features for specific modeling frameworks.

### Core Functionality

We've built a flexible foundation for model assessment that works with any probabilistic programming language through ArviZ's InferenceData format:

- **LOO-CV Implementation**: Leave-one-out cross-validation with multiple importance sampling methods (PSIS, SIS, TIS), comprehensive diagnostics, and flexible output scales.
- **WAIC Implementation**: Widely Applicable Information Criterion as an alternative approach to model assessment, with consistent interface and output formats.
- **Efficient Subsampling**: Statistical subsampling techniques for large datasets that reduce computation time while maintaining accuracy:
  - Multiple approximation methods (PLPD, LPD, TIS, SIS)
  - Various estimator approaches (diff_srs, hh_pps, srs)
  - Enhanced support for multidimensional data
- **Supporting Utilities**:
  - Pointwise LOO-CV calculations
  - Expected LOO computations
  - Various importance sampling methods

### Advanced PyMC Integration

Specialized tools for PyMC models that handle the complexities of PyMC's internal structures:

- **Universal PyMC Wrapper**: Standardized interface to model components that manages parameter transformations, data manipulation, and posterior sampling.
- **Re-LOO Implementation**: Exact refitting for problematic observations in LOO-CV when importance sampling fails to provide reliable estimates.
- **K-fold Cross-validation**: Comprehensive K-fold CV with customizable fold creation, stratified sampling, and detailed diagnostics.

### Maintenance and Dependencies

Our core dependencies include:
- numpy>=1.20.0
- scipy>=1.7.0
- pandas>=1.3.0
- arviz>=0.15.0
- pymc>=5.0.0

### Documentation

Expanded documentation covering all major features:

- **Usage Examples**:
  - Standard LOO-CV with different importance sampling methods
  - Subsampled LOO-CV with various approximation techniques
  - Integration examples with ArviZ
  - PyMC-specific workflows for reloo and k-fold cross-validation
- **API Documentation**: Comprehensive coverage of all key components, including importance sampling methods, subsampling estimators, and diagnostic tools.

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
