#!/bin/bash

set -e

# SDK version
: "${SDK_VER:=11.1.0}"

# docker tags
BASE_DOCKER_TAG=audioai-base:${SDK_VER}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if base image exists locally
if docker image inspect "$BASE_DOCKER_TAG" >/dev/null 2>&1; then
    echo "Base image $BASE_DOCKER_TAG already exists locally, skipping base build..."
else
    echo "Base image $BASE_DOCKER_TAG not found locally, building base image..."
    "${SCRIPT_DIR}/docker_build_base.sh"
fi

# Always build TI-specific image
echo "Building TI-specific image..."
"${SCRIPT_DIR}/docker_build_ti.sh"

echo "Docker build completed successfully!"

