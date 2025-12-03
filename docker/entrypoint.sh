#!/bin/bash
set -e

# setup proxy as required
source /root/setup_proxy.sh

# arch
echo "$(arch)"

# Ubuntu version
DISTRO_VER=$(lsb_release -r | cut -f2)
echo "DISTRO_VER=${DISTRO_VER}"
echo "$BASE_IMAGE"

# Get the IP address from eth0 interface
TARGET_IP=$(ifconfig eth0 | grep 'inet ' | awk '{print $2}')
if [ -z "$TARGET_IP" ]; then
    echo "Warning: Unable to determine the IP address of eth0 interface."
else
    echo "TARGET_IP=${TARGET_IP}"
fi
export TARGET_IP

# working dir
cd $WORK_DIR

# Set Numba cache directory for system Python (3.12)
export NUMBA_CACHE_DIR="/root/.numba_cache/python3.12"

exec "$@"
