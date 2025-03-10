# Change Log

## Unreleased

PyLoo is a comprehensive toolkit for model comparison and validation in Bayesian inference. Our latest developments focus on both core functionality and advanced features for specific modeling frameworks, specifically PyMC.

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
- **Moment Matching**: Computational alternative to model re-fitting that transforms posterior draws to better approximate leave-one-out posteriors, improving reliability of LOO-CV estimates for observations with high Pareto k diagnostics.

### Maintenance and Dependencies

Our core dependencies include:
- numpy>=1.20.0
- scipy>=1.7.0
- pandas>=1.3.0
- arviz>=0.15.0
- pymc>=5.0.0
- xarray>=2025.1.2
- pytensor>=2.27.1
