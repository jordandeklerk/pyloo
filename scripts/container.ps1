# PowerShell script for Docker operations

# Configuration
$IMAGE_NAME = "pyloo"
$CONTAINER_NAME = "pyloo-dev"

# Help message
function Show-Help {
    Write-Host "Usage: .\container.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build    Build the Docker image"
    Write-Host "  run      Run the Docker container"
    Write-Host "  test     Run tests in the container"
    Write-Host "  clean    Remove the container and image"
    Write-Host "  help     Show this help message"
}

# Build the Docker image
function Build-Image {
    Write-Host "Building Docker image..."
    docker build -t $IMAGE_NAME -f scripts/Dockerfile .
}

# Run the container
function Run-Container {
    Write-Host "Running container..."
    docker run --rm -it --name $CONTAINER_NAME $IMAGE_NAME
}

# Run tests in the container
function Test-Container {
    Write-Host "Running tests..."
    docker run --rm --name $CONTAINER_NAME $IMAGE_NAME pytest
}

# Clean up
function Clean-Environment {
    Write-Host "Cleaning up..."
    docker rm -f $CONTAINER_NAME 2>$null
    docker rmi -f $IMAGE_NAME 2>$null
}

# Main script
switch ($args[0]) {
    "build" { Build-Image }
    "run" { Run-Container }
    "test" { Test-Container }
    "clean" { Clean-Environment }
    "help" { Show-Help }
    default {
        Write-Host "Unknown command: $($args[0])"
        Show-Help
        exit 1
    }
}