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
ARG BASE_URL_VA
ARG BASE_URL_RT
ARG SDK_VER_MAJOR

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
ARG BASE_URL_VA
ARG BASE_URL_RT
ARG SDK_VER_MAJOR

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

# Python 3.10 and venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install vision_apps lib
WORKDIR /root/scripts
ADD install_vision_apps_lib.sh ./
RUN ./install_vision_apps_lib.sh

# install OS-RT libs
ADD install_osrt_libs.sh ./
RUN ./install_osrt_libs.sh

# install TIDL runtime libs
ADD install_tidl_libs.sh ./
RUN ./install_tidl_libs.sh

# install ti-rpmsg-char
WORKDIR /opt
RUN apt-get update && \
    apt-get install -y --no-install-recommends libtool autoconf automake build-essential && \
    git clone git://git.ti.com/rpmsg/ti-rpmsg-char.git --branch $RPMSG_LIB_VER --depth 1 --single-branch && \
    cd /opt/ti-rpmsg-char && \
    autoreconf -i && ./configure --host=aarch64-none-linux-gnu --prefix=/usr && \
    make && make install && \
    rm -rf /opt/ti-rpmsg-char && \
    rm -rf /var/lib/apt/lists/*

# Additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Setup Python virtual environment for GTCRN
COPY requirements_gtcrn.txt /tmp/requirements_gtcrn.txt
RUN mkdir -p /root/venv && \
    python3 -m venv /root/venv/gtcrn && \
    /root/venv/gtcrn/bin/pip install --upgrade pip wheel && \
    /root/venv/gtcrn/bin/pip install --no-cache-dir -r /tmp/requirements_gtcrn.txt && \
    rm /tmp/requirements_gtcrn.txt

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

ENV WORK_DIR=/root/audioai-modelzoo

# setup entrypoint
ENTRYPOINT ["/root/entrypoint.sh"]
