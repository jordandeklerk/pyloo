# Change Log

## Unreleased

### New features
- Added comprehensive PSIS diagnostic tools and visualizations in `diagnostics.py`
- Added base class for ELPD calculations in `elpd.py`
- Added ESS functions with ArviZ adaptations in `ess.py`
- Added multiple importance sampling methods:
  - Standard Importance Sampling (SIS) implementation in `sis.py`
  - Truncated Importance Sampling (TIS) implementation in `tis.py`
  - Unified importance sampling interface in `importance_sampling.py`
- Added helper utilities adapted from ArviZ in `utils.py`
- Enhanced PSIS test suite:
  - Added comprehensive test coverage for multidimensional arrays
  - Added tests for extreme value handling and high Pareto k values
  - Added parametrized tests for GPD inverse function
  - Added test cases matching ArviZ's test coverage

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
