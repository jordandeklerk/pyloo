#!/bin/bash

# Exit on error
set -e

# Configuration
IMAGE_NAME="pyloo"
CONTAINER_NAME="pyloo-dev"

# Help message
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build    Build the Docker image"
    echo "  run      Run the Docker container"
    echo "  test     Run tests in the container"
    echo "  clean    Remove the container and image"
    echo "  help     Show this help message"
}

# Build the Docker image
build() {
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME -f scripts/Dockerfile .
}

# Run the container
run() {
    echo "Running container..."
    docker run --rm -it --name $CONTAINER_NAME $IMAGE_NAME
}

# Run tests in the container
test() {
    echo "Running tests..."
    docker run --rm --name $CONTAINER_NAME $IMAGE_NAME pytest
}

# Clean up
clean() {
    echo "Cleaning up..."
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    docker rmi -f $IMAGE_NAME 2>/dev/null || true
}

# Main script
case "$1" in
    build)
        build
        ;;
    run)
        run
        ;;
    test)
        test
        ;;
    clean)
        clean
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac