#!/bin/bash
# Launch Jupyter Lab server accessible on TARGET_IP (env variable established inside the container)
[ ! -f "/.dockerenv" ] && echo "Error: Must run inside Docker container" && exit 1
[ -z "$TARGET_IP" ] && echo "Error: TARGET_IP not set" && exit 1

TOKEN="tidl"
echo "Access URL: http://$TARGET_IP:8888/lab?token=$TOKEN"
echo "Starting Jupyter Lab server..."
echo ""

# Clean previous Jupyter Lab workspace state
# rm -rf ~/.jupyter/lab/workspaces/*

jupyter-lab \
    --ip=$TARGET_IP \
    --no-browser \
    --allow-root \
    --IdentityProvider.token="$TOKEN" \
    --ServerApp.terminado_settings='{"shell_command": ["/bin/bash", "-l"]}'
    