#!/bin/bash

# Source SDK version
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${SCRIPT_DIR}/../VERSION"

# ARCH: arm64
ARCH=arm64

# base image
: "${BASE_IMAGE:=arm64v8/ubuntu:24.04}"

# docker tag
DOCKER_TAG=audioai:${SDK_VER}-${SOC}
echo "DOCKER_TAG = $DOCKER_TAG"

if [ "$#" -lt 1 ]; then
    CMD=/bin/bash
else
    CMD="$@"
fi

docker run -it --rm \
    -v /root/tidl/audioai-modelzoo:/root/tidl/audioai-modelzoo \
    -v /dev:/dev \
    --privileged \
    --network host \
    --env USE_PROXY=$USE_PROXY \
    --env ARCH=$ARCH \
    --env BASE_IMAGE=$BASE_IMAGE \
    --env SDK_VER=$SDK_VER \
    --env SOC=$SOC \
    --device-cgroup-rule='c 235:* rmw' \
    $DOCKER_TAG $CMD

