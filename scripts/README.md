# Scripts Directory

Various scripts and configuration files for building, testing, and maintaining the pyloo package.

## Files

### Docker-related Files
- `Dockerfile`: Configuration file for building a Docker container with the pyloo package installed.
- `container.sh`: Shell script for building and running the Docker container on Unix-like systems.
- `container.ps1`: PowerShell script for building and running the Docker container on Windows.

### Development Environment Setup
- `install_miniconda.sh`: Script to install Miniconda on your system. Detects your OS and architecture to install the appropriate version.
- `create_testenv.sh`: Creates a conda environment with all necessary dependencies for development and testing.
  - Supports specifying Python version and core dependency versions (numpy, scipy, pandas)
  - Installs all requirements from requirements*.txt files
  - Installs the package in development mode

### Code Quality
- `lint.sh`: Script to run all code quality checks
  - black for code formatting
  - isort for import sorting
  - flake8 for style guide enforcement
  - mypy for type checking
  - Supports both checking (`--check`) and fixing (`--fix`) modes

## Usage

### Setting Up Development Environment
1. Install Miniconda (if not already installed):
```bash
./install_miniconda.sh
```

2. Create test environment:
```bash
./create_testenv.sh
```
You can specify Python version:
```bash
PYTHON_VERSION=3.8 ./create_testenv.sh
```

### Running Code Quality Checks
Check code quality:
```bash
./lint.sh --check
```

Fix formatting issues:
```bash
./lint.sh --fix
```

### Using Docker
Build container:
```bash
./container.sh build  # Unix/macOS
.\container.ps1 build  # Windows
```

Run container:
```bash
./container.sh run  # Unix/macOS
.\container.ps1 run  # Windows
```

Run tests in container:
```bash
./container.sh test  # Unix/macOS
.\container.ps1 test  # Windows