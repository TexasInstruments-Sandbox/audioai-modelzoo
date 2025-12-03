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

ARG ARCH
ARG BASE_IMAGE
ARG USE_PROXY
ARG HTTP_PROXY
ARG DEBIAN_FRONTEND=noninteractive
ARG TIVA_LIB_VER
ARG RPMSG_LIB_VER
ARG SOC_NAME

#=========================================================================
FROM --platform=linux/${ARCH} ${BASE_IMAGE} AS base-0

#=========================================================================
FROM base-0 AS base-1
ARG USE_PROXY
ENV USE_PROXY=${USE_PROXY}
ARG HTTP_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTP_PROXY}

#=========================================================================
FROM base-${USE_PROXY}
ARG ARCH
ARG BASE_IMAGE
ARG DEBIAN_FRONTEND
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ARG TIVA_LIB_VER
ARG RPMSG_LIB_VER
ARG SOC_NAME
ENV SOC=$SOC_NAME

# setup proxy settings
ADD setup_proxy.sh /root/
ADD proxy /root/proxy
RUN /root/setup_proxy.sh

# tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    corkscrew \
    vim \
    tmux \
    net-tools \
    lsb-release \
    cmake && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Python 3 (Ubuntu 24.04 uses Python 3.12) and venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python 3.10 for torch==1.11.0 compatibility (torch 1.11 doesn't support Python 3.12)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install vision_apps lib and DL runtime libs
COPY lib/*  /usr/lib/
COPY lib_gstreamer-1.0/* /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
COPY include/processor_sdk /usr/include/processor_sdk
RUN ln -s /usr/lib/libtivision_apps.so.$TIVA_LIB_VER /usr/lib/libtivision_apps.so && \
    ln -s /usr/lib/libtidl_tfl_delegate.so.1.0 /usr/lib/libtidl_tfl_delegate.so && \
    ln -s /usr/lib/libtidl_onnxrt_EP.so.1.0 /usr/lib/libtidl_onnxrt_EP.so && \
    ln -s /usr/lib/libvx_tidl_rt.so.1.0 /usr/lib/libvx_tidl_rt.so && \
    ln -s /usr/lib/libti_rpmsg_char.so.$RPMSG_LIB_VER /usr/lib/libti_rpmsg_char.so.0 && \
    ln -s /usr/lib/libti_rpmsg_char.so.0 /usr/lib/libti_rpmsg_char.so

# Additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt --break-system-packages && \
    rm /tmp/requirements.txt

# Warmup system Python packages (Numba, PyTorch, etc.) - Python 3.12
COPY warmup_packages.py /tmp/warmup_packages.py
RUN mkdir -p /root/.numba_cache/python3.12 && \
    NUMBA_CACHE_DIR=/root/.numba_cache/python3.12 python3 /tmp/warmup_packages.py && \
    rm /tmp/warmup_packages.py

# Setup Python virtual environment for GTCRN (using Python 3.10 for torch==1.11.0 compatibility)
COPY requirements_gtcrn.txt /tmp/requirements_gtcrn.txt
RUN mkdir -p /root/venv && \
    python3.10 -m venv /root/venv/gtcrn && \
    /root/venv/gtcrn/bin/pip install --upgrade pip wheel setuptools && \
    /root/venv/gtcrn/bin/pip install --no-cache-dir -r /tmp/requirements_gtcrn.txt && \
    rm /tmp/requirements_gtcrn.txt

# Warmup GTCRN venv packages - Python 3.10
COPY warmup_packages.py /tmp/warmup_packages.py
RUN mkdir -p /root/.numba_cache/python3.10 && \
    NUMBA_CACHE_DIR=/root/.numba_cache/python3.10 /root/venv/gtcrn/bin/python /tmp/warmup_packages.py && \
    rm /tmp/warmup_packages.py

# Configure GTCRN venv to use separate Numba cache
RUN echo 'export NUMBA_CACHE_DIR="/root/.numba_cache/python3.10"' >> /root/venv/gtcrn/bin/activate

#=========================================================================
# add scripts
ADD entrypoint.sh /root/

## .profile and .bashrc
WORKDIR /root
RUN echo "if [ -n \"$BASH_VERSION\" ]; then"     >  .profile && \
    echo "    # include .bashrc if it exists"    >> .profile && \
    echo "    if [ -f \"$HOME/.bashrc\" ]; then" >> .profile && \
    echo "        . \"$HOME/.bashrc\""           >> .profile && \
    echo "    fi"                                >> .profile && \
    echo "fi"                                    >> .profile && \
    echo "#!/bin/bash"                           >  .bashrc  && \
    echo "export PS1=\"${debian_chroot:+($debian_chroot)}\u@docker:\w\$ \"" >> .bashrc

# add label
LABEL TI_IMAGE_SOURCE=${BASE_IMAGE}

ENV WORK_DIR=/root/tidl/audioai-modelzoo

# setup entrypoint
ENTRYPOINT ["/root/entrypoint.sh"]
