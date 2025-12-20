#!/bin/bash
#
# Jupyter Lab Launch Script for AudioAI ModelZoo
#
# This script launches Jupyter Lab server with:
# - Simple token authentication (token='tidl')
# - Auto-opens three inference notebooks in tabs
# - Disables news notification popup
#
# Usage: ./jupyter_lab.sh (must run inside Docker container)
#

# Launch Jupyter Lab server accessible on TARGET_IP (env variable established inside the container)
[ ! -f "/.dockerenv" ] && echo "Error: Must run inside Docker container" && exit 1
[ -z "$TARGET_IP" ] && echo "Error: TARGET_IP not set" && exit 1

TOKEN="tidl"
echo -e "\033[1;32m================================================================\033[0m"
echo -e "\033[1;33mAccess URL: http://$TARGET_IP:8888/lab?token=$TOKEN\033[0m"
echo -e "\033[1;32m================================================================\033[0m"
echo "Starting Jupyter Lab server. Please wait..."
echo ""

# Create workspace configuration to auto-open the three notebooks
WORKSPACE_DIR=~/.jupyter/lab/workspaces
mkdir -p "$WORKSPACE_DIR"

# Jupyter generates the hash from: "file://" + workspace_root
# For /root/tidl/audioai-modelzoo/inference, the hash is 37a8
WORKSPACE_FILE="$WORKSPACE_DIR/default-37a8.jupyterlab-workspace"

cat > "$WORKSPACE_FILE" << 'EOF'
{
  "data": {
    "layout-restorer:data": {
      "main": {
        "dock": {
          "type": "tab-area",
          "currentIndex": 0,
          "widgets": [
            "notebook:gtcrn_se/gtcrn_inference.ipynb",
            "notebook:vggish11_sc/vggish_inference.ipynb",
            "notebook:yamnet_sc/yamnet_inference.ipynb"
          ]
        },
        "current": "notebook:gtcrn_se/gtcrn_inference.ipynb"
      },
      "down": {
        "size": 0,
        "widgets": []
      },
      "left": {
        "collapsed": false,
        "visible": true,
        "current": "filebrowser",
        "widgets": [
          "filebrowser",
          "running-sessions",
          "@jupyterlab/toc:plugin",
          "extensionmanager.main-view"
        ]
      },
      "right": {
        "collapsed": true,
        "visible": true,
        "widgets": [
          "jp-property-inspector",
          "debugger-sidebar"
        ]
      },
      "relativeSizes": [0.2, 0.8, 0],
      "top": {
        "simpleVisibility": true
      }
    },
    "notebook:gtcrn_se/gtcrn_inference.ipynb": {
      "data": {
        "path": "gtcrn_se/gtcrn_inference.ipynb",
        "factory": "Notebook"
      }
    },
    "notebook:vggish11_sc/vggish_inference.ipynb": {
      "data": {
        "path": "vggish11_sc/vggish_inference.ipynb",
        "factory": "Notebook"
      }
    },
    "notebook:yamnet_sc/yamnet_inference.ipynb": {
      "data": {
        "path": "yamnet_sc/yamnet_inference.ipynb",
        "factory": "Notebook"
      }
    }
  },
  "metadata": {
    "id": "default"
  }
}
EOF

# Disable news notifications
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
cat > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/notification.jupyterlab-settings << 'EOF'
{
  "fetchNews": "false"
}
EOF

# Launch Jupyter Lab
jupyter-lab \
    --ip=$TARGET_IP \
    --no-browser \
    --allow-root \
    --IdentityProvider.token="$TOKEN" \
    --ServerApp.terminado_settings='{"shell_command": ["/bin/bash", "-l"]}'
    