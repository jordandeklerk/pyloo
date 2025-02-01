# pyloo Examples

Template directory containing future examples demonstrating how to use pyloo for Leave-One-Out Cross-Validation (LOO-CV) and Pareto Smoothed Importance Sampling (PSIS) with various Bayesian models and frameworks.

## Directory Structure

```
examples/
├── basic/                # Basic usage examples
│   ├── basic_loo.py      # Basic LOO-CV example
│   └── basic_psis.py     # Basic PSIS example
├── frameworks/           # Framework-specific examples
│   ├── pymc/             # PyMC examples
│   ├── pystan/           # PyStan examples
│   └── numpyro/          # NumPyro examples
│   └── tensorflow/       # Tensorflow examples
└── advanced/             # Advanced usage examples
    ├── moment_match/     # Moment matching examples
    └── k_fold/           # K-fold cross-validation examples
```

## Running the Examples

Each example is self-contained and can be run directly. Make sure you have pyloo installed:

```bash
pip install pyloo
```

For framework-specific examples, you'll need to install the relevant frameworks:

```bash
# For PyMC examples
pip install pymc

# For PyStan examples
pip install pystan

# For NumPyro examples
pip install numpyro

# For Tensorflow examples
pip install tensorflow
```

## Example Descriptions

### Basic Examples
- `basic_loo.py`: Demonstrates basic LOO-CV computation
- `basic_psis.py`: Shows how to perform PSIS for model comparison

### Framework Examples
- PyMC examples show integration with PyMC models
- PyStan examples demonstrate usage with Stan models
- NumPyro examples illustrate usage with NumPyro models
- Tensorflow examples illustrate usage with NumPyro models

### Advanced Examples
- Moment matching examples for improved LOO-CV estimates
- K-fold cross-validation for larger datasets

## Contributing

If you have an example you'd like to contribute:
1. Follow the existing example structure
2. Include clear documentation and comments
3. Ensure the example is self-contained
4. Add any necessary requirements to the example's README
5. Submit a pull request

## Notes

- Examples are meant to be educational and demonstrate best practices
- Each example includes extensive comments explaining the process
- For more detailed documentation, see the main pyloo documentation