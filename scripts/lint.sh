#!/usr/bin/env bash

set -ex # fail on first error, print commands

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -f, --fix      Automatically fix issues where possible"
    echo "  -c, --check    Only check for issues without fixing"
}

# Default to check mode
FIX_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--fix)
            FIX_MODE=true
            shift
            ;;
        -c|--check)
            FIX_MODE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to run black
run_black() {
    if [ "$FIX_MODE" = true ]; then
        black .
    else
        black --check .
    fi
}

# Function to run isort
run_isort() {
    if [ "$FIX_MODE" = true ]; then
        isort .
    else
        isort --check-only .
    fi
}

# Function to run flake8
run_flake8() {
    flake8 .
}

# Function to run mypy
run_mypy() {
    mypy .
}

echo "Running linting tools..."

# Run black for code formatting
echo "Running black..."
run_black

# Run isort for import sorting
echo "Running isort..."
run_isort

# Run flake8 for style guide enforcement
echo "Running flake8..."
run_flake8

# Run mypy for type checking
echo "Running mypy..."
run_mypy

echo "Linting completed!"