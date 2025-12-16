#!/bin/bash

set -e

# Source SDK version
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source "${SCRIPT_DIR}/sdk_version.sh"

# base image
: "${BASE_IMAGE:=arm64v8/ubuntu:24.04}"

# build arguments - detect from filesystem, fallback to defaults
TIVA_LIB_VER=$(ls -1 /usr/lib/libtivision_apps.so.* 2>/dev/null | sed 's|.*/libtivision_apps.so.||' | sort -V | tail -1 )
RPMSG_LIB_VER=$(ls -1 /usr/lib/libti_rpmsg_char.so.* 2>/dev/null | sed 's|.*/libti_rpmsg_char.so.||' | sort -V | tail -1 )
echo "TIVA_LIB_VER = ${TIVA_LIB_VER}"
echo "RPMSG_LIB_VER = ${RPMSG_LIB_VER}"

# check version mismatch
if [ "${TIVA_LIB_VER}" != "${SDK_VER}" ]; then
    echo "WARNING: TIVA_LIB_VER (${TIVA_LIB_VER}) does not match SDK_VER (${SDK_VER})"
fi

# docker tags
BASE_DOCKER_TAG=audioai-base:${SDK_VER}
DOCKER_TAG=audioai:${SDK_VER}-${SOC}
echo "BASE_DOCKER_TAG = ${BASE_DOCKER_TAG}"
echo "DOCKER_TAG = ${DOCKER_TAG}"

# copy files to be added while docker-build
DST_DIR=.

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

# create tarballs for custom Python packages
mkdir -p ${DST_DIR}/python_packages
if [ -d "/usr/lib/python3.12/site-packages/onnxruntime" ]; then
    tar -czf ${DST_DIR}/python_packages/onnxruntime.tar.gz -C /usr/lib/python3.12/site-packages onnxruntime
fi
if [ -d "/usr/lib/python3.12/site-packages/tflite_runtime" ]; then
    tar -czf ${DST_DIR}/python_packages/tflite_runtime.tar.gz -C /usr/lib/python3.12/site-packages tflite_runtime
fi

# docker-build TI-specific image
SECONDS=0
docker build \
    -t $DOCKER_TAG \
    --build-arg BASE_IMAGE_TAG=$BASE_DOCKER_TAG \
    --build-arg TIVA_LIB_VER=$TIVA_LIB_VER \
    --build-arg RPMSG_LIB_VER=$RPMSG_LIB_VER \
    --build-arg SOC_NAME=$SOC \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    -f Dockerfile.ti .
echo "Docker build -t $DOCKER_TAG completed!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

rm -rf ${DST_DIR}/lib
rm -rf ${DST_DIR}/lib_gstreamer-1.0
rm -rf ${DST_DIR}/include
rm -rf ${DST_DIR}/python_packages
