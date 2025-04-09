# Change Log

## Unreleased

PyLOO is a comprehensive toolkit for model comparison and validation in Bayesian inference. Our latest developments focus on both core functionality and advanced features for specific modeling frameworks, specifically PyMC.

### Core Functionality

We've built a flexible foundation based on similar implementations in Arviz for model assessment that works with any probabilistic programming language through ArviZ's InferenceData format:

- **LOO-CV Implementation**: Leave-one-out cross-validation with multiple importance sampling methods (PSIS, SIS, TIS), comprehensive diagnostics, and flexible output scales.
- **WAIC Implementation**: Widely Applicable Information Criterion as an alternative model assessment metric (`waic`).
- **Non-factorized LOO-CV**: Compute LOO-CV for multivariate normal and Student-t models where the likelihood cannot be factorized by observations (`loo_nonfactor`).
- **Model Comparison**: Compare models based on ELPD using various methods like stacking (`loo_compare`).
- **LOO-Based Metrics**: Estimate predictive performance (e.g., MAE, MSE, CRPS) using LOO estimates (`loo_predictive_metric`, `loo_score`).
- **Grouped & Subsampled CV**: Perform Leave-One-Group-Out CV (`loo_group`) and efficient statistical subsampling techniques for large datasets (`loo_subsample`) that reduce computation time while maintaining accuracy:
  - Multiple approximation methods (`PLPD`, `LPD`)
  - Various estimator approaches (`diff_srs`, `hh_pps`, `srs`)
  - Enhanced support for multidimensional data
- **Supporting Utilities**:
  - Pointwise LOO-CV calculations
  - Expected LOO computations
  - Various importance sampling methods

### Advanced & PyMC Integration

Specialized tools for PyMC models that handle the complexities of PyMC's internal structures:

- **PyMC Model Interface**: Seamless integration with PyMC models for streamlined workflow.
- **Exact Refitting (Reloo)**: Refit models for problematic observations identified by LOO diagnostics (`reloo`).
- **K-Fold Cross-Validation**: Flexible K-fold CV implementation with stratification and diagnostics (`loo_kfold`).
- **Moment Matching**: Improve LOO-CV reliability for challenging observations (`loo_moment_match`).
- **Variational Inference Support**: Compute LOO-CV for models fitted with Laplace or ADVI approximations (`loo_approximate_posterior`).

### Maintenance and Dependencies

Our core dependencies include:
- numpy>=1.20.0
- scipy>=1.7.0
- pandas>=1.3.0
- arviz>=0.15.0
- pymc>=5.0.0
- xarray>=2025.1.2
- pytensor>=2.27.1
