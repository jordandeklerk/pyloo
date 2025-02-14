#!/usr/bin/env bash

set -ex # fail on first error, print commands

command -v conda >/dev/null 2>&1 || {
  echo "Requires conda but it is not installed.  Run install_miniconda.sh." >&2;
  exit 1;
}

# if no python specified, use 3.11 (as specified in readthedocs.yaml)
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
NUMPY_VERSION=${NUMPY_VERSION:-latest}
SCIPY_VERSION=${SCIPY_VERSION:-latest}
PANDAS_VERSION=${PANDAS_VERSION:-latest}

# Update Conda to include latest build channels
conda update conda
conda update setuptools

if [[ $* != *--global* ]]; then
    ENVNAME="testenv_${PYTHON_VERSION}_NUMPY_${NUMPY_VERSION}_SCIPY_${SCIPY_VERSION}_PANDAS_${PANDAS_VERSION}"

    if conda env list | grep -q ${ENVNAME}
    then
        echo "Environment ${ENVNAME} already exists, keeping up to date"
    else
        echo "Creating environment ${ENVNAME}"
        conda create -n ${ENVNAME} --yes pip python=${PYTHON_VERSION}
    fi

    # Activate environment immediately
    source activate ${ENVNAME}

    if [ "$DOCKER_BUILD" = true ] ; then
        # Also add it to root bash settings to set default if used later
        echo "Creating .bashrc profile for docker image"
        echo "set conda_env=${ENVNAME}" > /root/activate_conda.sh
        echo "source activate ${ENVNAME}" >> /root/activate_conda.sh
    fi
fi

# Install pyloo dependencies
pip install --upgrade pip wheel

# Install core dependencies with version control
if [ "$NUMPY_VERSION" = "latest" ]; then
    pip --no-cache-dir install numpy
else
    pip --no-cache-dir install numpy==${NUMPY_VERSION}
fi

if [ "$SCIPY_VERSION" = "latest" ]; then
    pip --no-cache-dir install scipy
else
    pip --no-cache-dir install scipy==${SCIPY_VERSION}
fi

if [ "$PANDAS_VERSION" = "latest" ]; then
    pip --no-cache-dir install pandas
else
    pip --no-cache-dir install pandas==${PANDAS_VERSION}
fi

# Install all requirements
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r requirements-dev.txt
pip install --no-cache-dir -r requirements-test.txt
pip install --no-cache-dir -r requirements-docs.txt

# Install the package in development mode
pip install -e .
