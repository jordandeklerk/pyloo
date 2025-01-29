# Contributing to pyloo

Thank you for your interest in contributing to pyloo! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/pyloo.git
cd pyloo
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install development dependencies:
```bash
pip install -e ".[dev,test,docs]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

1. Create a new branch for your feature or bugfix:
```bash
git checkout -b feature-name
```

2. Make your changes, following our coding standards:
- Use Black for code formatting
- Sort imports with isort
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed

3. Run the test suite:
```bash
pytest
```

4. Build and check documentation:
```bash
cd docs
make html
```

5. Commit your changes:
```bash
git add .
git commit -m "Description of changes"
```

## Pull Request Process

1. Update your fork to include the latest changes from the main repository
2. Push your changes to your fork
3. Create a Pull Request through GitHub
4. Ensure the CI checks pass
5. Wait for review and address any feedback

## Testing

- Write tests for all new functionality
- Tests should be placed in the `tests/` directory
- Run the full test suite before submitting a PR
- Aim for high test coverage of new code

## Documentation

- Update docstrings for any new or modified functions/classes
- Follow NumPy docstring format
- Update relevant documentation in the `docs/` directory
- Include doctest examples where appropriate

## Code Style

We use several tools to maintain code quality:
- Black for code formatting
- isort for import sorting
- flake8 for style guide enforcement
- mypy for type checking

The pre-commit hooks will automatically check these when you commit.

## Questions or Problems?

Feel free to open an issue on GitHub if you have any questions or problems.