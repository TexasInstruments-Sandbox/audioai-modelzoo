#!/bin/bash

set -e

# Source SDK version
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${SCRIPT_DIR}/../VERSION"

# ARCH
ARCH=arm64

# base image
: "${BASE_IMAGE:=arm64v8/ubuntu:24.04}"

# docker tags
BASE_DOCKER_TAG=audioai-base:${SDK_VER}
echo "BASE_DOCKER_TAG = ${BASE_DOCKER_TAG}"

# for TI proxy network settings
: "${USE_PROXY:=0}"

# modify the server and proxy URLs as required
if [ "${USE_PROXY}" -ne "0" ]; then
    HTTP_PROXY=http://webproxy.ext.ti.com:80
fi
echo "USE_PROXY = $USE_PROXY"

# copy files to be added while docker-build
# requirement: git-pull edgeai-ti-proxy repo and source edgeai-ti-proxy/setup_proxy.sh
DST_DIR=.
mkdir -p $DST_DIR/proxy
PROXY_DIR=$HOME/proxy
if [[ "$(arch)" == "aarch64" && "$(whoami)" == "root" ]]; then
    PROXY_DIR=/opt/proxy
fi
if [ -d "$PROXY_DIR" ]; then
    cp -rp $PROXY_DIR/* ${DST_DIR}/proxy
fi

# docker-build base image
SECONDS=0
docker build \
    -t $BASE_DOCKER_TAG \
    --build-arg ARCH=$ARCH \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg USE_PROXY=$USE_PROXY \
    --build-arg HTTP_PROXY=$HTTP_PROXY \
    -f Dockerfile.base .
echo "Docker build -t $BASE_DOCKER_TAG completed!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

rm -rf ${DST_DIR}/proxy
