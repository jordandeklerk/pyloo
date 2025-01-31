# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup
- Basic package structure
- Development environment configuration
- Core dependencies:
  - numpy>=1.20.0
  - scipy>=1.7.0
  - pandas>=1.3.0
- Enhanced PSIS test suite:
  - Added comprehensive test coverage for multidimensional arrays
  - Added tests for extreme value handling and high Pareto k values
  - Added parametrized tests for GPD inverse function
  - Added test cases matching ArviZ's test coverage
- Updated test dependencies:
  - Added arviz>=0.16.0 for test compatibility
  - Added numpy>=1.21.0 and scipy>=1.7.0 as explicit test requirements

## v[0.1.0] (2025-01-29)

### Added
- Project initialization
- Basic repository structure
- Development tooling setup:
  - pytest for testing
  - black for code formatting
  - isort for import sorting
  - flake8 for linting
  - mypy for type checking
  - pre-commit hooks

[Unreleased]: https://github.com/your-username/pyloo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/pyloo/releases/tag/v0.1.0