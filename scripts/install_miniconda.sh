#!/usr/bin/env bash

set -ex # fail on first error, print commands

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ $(uname -m) == 'arm64' ]]; then
        # M1/M2 Mac
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    else
        # Intel Mac
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Directory for installation
CONDA_DIR=$HOME/miniconda3

# Download and install Miniconda
if [ ! -d "$CONDA_DIR" ]; then
    echo "Downloading Miniconda..."
    wget $MINICONDA_URL -O miniconda.sh

    echo "Installing Miniconda..."
    bash miniconda.sh -b -p $CONDA_DIR
    rm miniconda.sh

    # Add conda to path
    echo "Configuring Miniconda..."
    $CONDA_DIR/bin/conda init "$(basename "$SHELL")"

    echo "Miniconda installed successfully!"
    echo "Please restart your shell or run:"
    echo "    source ~/.$(basename "$SHELL")rc"
else
    echo "Miniconda is already installed in $CONDA_DIR"
fi

# Update conda
echo "Updating conda..."
source $CONDA_DIR/bin/activate
conda update -n base -c defaults conda -y
