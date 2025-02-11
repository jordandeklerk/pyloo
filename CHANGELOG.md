# Change Log

## Unreleased

### New features
- Added base class for ELPD calculations in `elpd.py`
- Added ESS functions with ArviZ adaptations in `ess.py`
- Added multiple importance sampling methods:
  - Standard Importance Sampling (SIS) implementation in `sis.py`
  - Truncated Importance Sampling (TIS) implementation in `tis.py`
  - Unified importance sampling interface in `importance_sampling.py`
  - PSIS implementation in `psis.py`
- Added helper utilities adapted from ArviZ in `utils.py`
- Implemented standard LOO-CV computation in `loo.py`:
  - Support for multiple importance sampling methods (PSIS, SIS, TIS)
  - Comprehensive error handling and diagnostics
  - Integration with ArviZ for data handling
  - Flexible output scales (log, negative_log, deviance)
- Implemented efficient approximate LOO-CV using subsampling in `loo_subsample.py`:
  - Multiple approximation methods (PLPD, LPD, TIS, SIS)
  - Various estimator methods (diff_srs, hh_pps, srs)
  - Enhanced support for multidimensional data
  - Additional diagnostics and error metrics
  - Efficient computation for large datasets
- Added pointwise LOO-CV calculations in `loo_i.py`
- Added expected LOO computations in `e_loo.py`

### Maintenance and fixes
- Updated test dependencies:
  - Updated arviz to version 0.20.0 for test compatibility
  - Added numpy>=1.21.0 and scipy>=1.7.0 as explicit test requirements
- Core dependencies:
  - numpy>=1.20.0
  - scipy>=1.7.0
  - pandas>=1.3.0

### Documentation
- Basic package structure and documentation setup
- Development environment configuration
- Added comprehensive usage examples for LOO-CV implementations:
  - Standard LOO-CV with different importance sampling methods
  - Subsampled LOO-CV with various approximation techniques
  - Integration examples with ArviZ
- Added detailed API documentation for:
  - Importance sampling methods (PSIS, SIS, TIS)
  - Subsampling estimators (diff_srs, hh_pps, srs)
  - Diagnostic tools and warnings

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
