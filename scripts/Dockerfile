FROM conda/miniconda3

LABEL maintainer="Jordan Deklerk"

ARG PYTHON_VERSION=3.11
ARG NUMPY_VERSION=latest
ARG SCIPY_VERSION=latest
ARG PANDAS_VERSION=latest

ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV NUMPY_VERSION=${NUMPY_VERSION}
ENV SCIPY_VERSION=${SCIPY_VERSION}
ENV PANDAS_VERSION=${PANDAS_VERSION}

# Change behavior of create_testenv.sh script
ENV DOCKER_BUILD=true

# For Sphinx documentation builds
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Update container and install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    pandoc \
    vim \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/pyloo

# Copy requirements and scripts
COPY requirements*.txt ./
COPY scripts ./scripts
RUN find . -type f -print0 | xargs -0 dos2unix

# Clear any cached files
RUN find -type d -name __pycache__ -exec rm -rf {} +
RUN find -type d -name "*.egg-info" -exec rm -rf {} +

# Create conda environment using our script
RUN ./scripts/create_testenv.sh

# Set automatic conda activation
ENV BASH_ENV="/root/activate_conda.sh"
RUN echo ". /root/activate_conda.sh" > /root/.bashrc

# Remove conda cache
RUN conda clean --all

# Copy the rest of the package
COPY . .
RUN find . -type f -print0 | xargs -0 dos2unix

# Clear any cached files again after full copy
RUN find -type d -name __pycache__ -exec rm -rf {} +
RUN find -type d -name "*.egg-info" -exec rm -rf {} +

# Install package in development mode
RUN /bin/bash -c "source /root/.bashrc && python -m pip install -e ."

# Default command to verify installation
CMD ["python", "-c", "import pyloo; print(f'pyloo version {pyloo.__version__} installed successfully')"]
