# Scripts Directory

This directory contains scripts and configuration files for building and deploying the pyloo package.

## Files

- `Dockerfile`: Configuration file for building a Docker container with the pyloo package installed.
- `container.sh`: Shell script for building and running the Docker container on Unix-like systems.
- `container.ps1`: PowerShell script for building and running the Docker container on Windows.

## Docker Build

The Docker configuration provides a reproducible environment for running pyloo. The Dockerfile:
- Uses Python 3.11 as the base image
- Installs all required dependencies
- Installs the pyloo package in development mode
- Sets up a working environment for using the package

### Building the Container

Unix/macOS:
```bash
./container.sh build
```

Windows:
```powershell
.\container.ps1 build
```

### Running the Container

Unix/macOS:
```bash
./container.sh run
```

Windows:
```powershell
.\container.ps1 run