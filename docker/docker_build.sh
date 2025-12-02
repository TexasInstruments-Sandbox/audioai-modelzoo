#!/bin/bash

#  Copyright (C) 2025 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -e

# ARCH
ARCH=arm64

# base image
: "${BASE_IMAGE:=arm64v8/ubuntu:24.04}"

# SDK version
: "${SDK_VER:=11.1.0}"

# build arguments
TIVA_LIB_VER=11.1.0
RPMSG_LIB_VER=0.6.9

# docker tag
DOCKER_TAG=audioai:${SDK_VER}
echo "DOCKER_TAG = $DOCKER_TAG"

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

# copy Processor SDK libraries to the temporary folder
mkdir -p ${DST_DIR}/lib
Lib_files=(
    /usr/lib/libtivision_apps.so.${TIVA_LIB_VER}
    /usr/lib/libti_rpmsg_char.so.${RPMSG_LIB_VER}
    /usr/lib/libtidl_tfl_delegate.so.1.0
    /usr/lib/libtidl_onnxrt_EP.so.1.0
    /usr/lib/libvx_tidl_rt.so.1.0
)
for Lib_file in ${Lib_files[@]}; do
    cp $Lib_file ${DST_DIR}/lib
done

# copy a GST lib that was updated from PSDK
mkdir -p ${DST_DIR}/lib_gstreamer-1.0
cp /usr/lib/gstreamer-1.0/libgstvideo4linux2.so ${DST_DIR}/lib_gstreamer-1.0

# copy PSDK header files
mkdir -p ${DST_DIR}/include
cp -rp /usr/include/processor_sdk ${DST_DIR}/include

# docker-build
SECONDS=0
docker build \
    -t $DOCKER_TAG \
    --build-arg ARCH=$ARCH \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg USE_PROXY=$USE_PROXY \
    --build-arg HTTP_PROXY=$HTTP_PROXY \
    --build-arg TIVA_LIB_VER=$TIVA_LIB_VER \
    --build-arg RPMSG_LIB_VER=$RPMSG_LIB_VER \
    --build-arg SOC_NAME=$SOC \
    -f Dockerfile .
echo "Docker build -t $DOCKER_TAG completed!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

rm -rf ${DST_DIR}/proxy
rm -rf ${DST_DIR}/lib
rm -rf ${DST_DIR}/lib_gstreamer-1.0
rm -rf ${DST_DIR}/include

