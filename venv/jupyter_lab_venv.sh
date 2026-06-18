#!/bin/bash
#
# Jupyter Lab Launch Script for AudioAI ModelZoo — native venv (no Docker)
#
# Launches Jupyter Lab from the native .venv with:
# - Simple token authentication (token='tidl')
# - Auto-opens three inference notebooks in tabs
# - Disables news notification popup
#
# Usage (on the EVM, from the repo root or inference/):
#   ./venv/jupyter_lab_venv.sh
#
# Assumes the venv has been set up with ./venv/setup_venv.sh.
# Must run as root on the EVM (~/tidl/audioai-modelzoo/inference path must resolve
# to /root/tidl/audioai-modelzoo/inference for the JupyterLab workspace hash to match).
#

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
elif [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Error: venv not found at ${VENV_DIR}" >&2
    echo "       Run ./venv/setup_venv.sh first." >&2
    exit 1
fi

# Derive TARGET_IP from the primary network interface (interface-name-agnostic;
# the Docker entrypoint.sh hardcodes eth0 but the bare EVM may use end0/eth1)
if [ -z "$TARGET_IP" ]; then
    TARGET_IP=$(ip -4 -o addr show scope global up 2>/dev/null \
                | awk '{print $4}' | cut -d/ -f1 | head -n1)
fi
[ -z "$TARGET_IP" ] && TARGET_IP=0.0.0.0
export TARGET_IP

# SOC defaults to am62a — exported so the Jupyter kernel inherits it
# (notebooks read os.environ["SOC"] to locate TIDL artifacts)
export SOC="${SOC:-am62a}"

TOKEN="tidl"
echo -e "\033[1;32m================================================================\033[0m"
echo -e "\033[1;33mAccess URL: http://$TARGET_IP:8888/lab?token=$TOKEN\033[0m"
echo -e "\033[1;32m================================================================\033[0m"
echo "Starting Jupyter Lab server. Please wait..."
echo ""

# Change to inference/ so the JupyterLab workspace hash (37a8) is valid.
# Hash 37a8 is derived from "file://" + /root/tidl/audioai-modelzoo/inference
cd "${REPO_DIR}/inference"

# Create workspace configuration to auto-open the three notebooks
WORKSPACE_DIR=~/.jupyter/lab/workspaces
mkdir -p "$WORKSPACE_DIR"

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
