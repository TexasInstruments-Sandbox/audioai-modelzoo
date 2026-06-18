#!/bin/bash
#
# One-time native (no-Docker) venv setup for AudioAI ModelZoo on AM62A EVM.
#
# Creates .venv/ in the repo root with --system-site-packages so the TIDL-enabled
# onnxruntime and tflite_runtime from the Yocto rootfs (/usr/lib/python3.12/site-packages)
# are accessible, while standard deps (torch, torchaudio, etc.) are pip-installed
# directly into the venv. numpy<2 is pinned to stay ABI-compatible with the
# system-built TIDL onnxruntime.
#
# Usage (on the EVM, from the repo root):
#   ./venv/setup_venv.sh
#
# Run via evm-run.sh so the EVM proxy is sourced (needed for PyPI / PyTorch index):
#   evm-run.sh "cd ~/tidl/audioai-modelzoo && ./venv/setup_venv.sh"
#
set -e

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
source "${REPO_DIR}/VERSION"

VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
SYS_PY=/usr/bin/python3

echo "=== AudioAI ModelZoo: native venv setup ==="
echo "Repo:  $REPO_DIR"
echo "Venv:  $VENV_DIR"
echo ""

# --------------------------------------------------------------------------
# 1. Sanity: TIDL onnxruntime must already exist in the system site-packages.
#    This script is intended to run on the AM62A EVM with the Processor SDK rootfs.
# --------------------------------------------------------------------------
SYS_SP="$(${SYS_PY} -c "
import site, os
matches = [p for p in site.getsitepackages() if os.path.isdir(os.path.join(p, 'onnxruntime'))]
if matches:
    print(matches[0])
" 2>/dev/null)"

if [ -z "$SYS_SP" ]; then
    echo "ERROR: system onnxruntime not found in any site-packages directory." >&2
    echo "       This script must run on the AM62A EVM with the Processor SDK rootfs." >&2
    echo "       (Expected: /usr/lib/python3.12/site-packages/onnxruntime)" >&2
    exit 1
fi
echo "System site-packages with TIDL onnxruntime: $SYS_SP"

# --------------------------------------------------------------------------
# 2. Create venv with --system-site-packages so the TIDL packages are visible.
#    numpy<2 pip-installed into the venv shadows the system numpy (venv site-packages
#    precede system paths in sys.path) while remaining ABI-compatible with the
#    system-built TIDL onnxruntime.
# --------------------------------------------------------------------------
echo ""
echo "--- Creating venv ---"
${SYS_PY} -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# --------------------------------------------------------------------------
# 3. Install standard deps from venv/requirements.txt (torch, torchaudio,
#    soundfile, matplotlib, notebook, jupyterlab, numpy<2).
#    onnxruntime is intentionally absent — it comes from the system rootfs.
# --------------------------------------------------------------------------
echo ""
echo "--- Installing dependencies (this may take several minutes) ---"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# --------------------------------------------------------------------------
# 4. Verify installation
# --------------------------------------------------------------------------
echo ""
echo "--- Verifying installation ---"

python -c "
import numpy, torch, torchaudio, soundfile, matplotlib, jupyterlab
print('  numpy          ', numpy.__version__)
print('  torch          ', torch.__version__)
print('  torchaudio     ', torchaudio.__version__)
print('  soundfile      ', soundfile.__version__)
print('  jupyterlab     OK')
"

python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('  onnxruntime    ', ort.__version__)
print('  providers      ', providers)
assert 'TIDLExecutionProvider' in providers, \
    'TIDLExecutionProvider not found! Check that system onnxruntime is the TIDL-enabled build.'
print('  TIDLExecutionProvider: OK')
"

python -c "import tflite_runtime; print('  tflite_runtime OK')" 2>/dev/null \
    || echo "  tflite_runtime not found (not required for current demos)"

# --------------------------------------------------------------------------
# 5. Hint about models/artifacts if not yet downloaded.
#    The gtcrn demo is CPU-only, but the vggish/yamnet (TIDL) demos load compiled
#    artifacts from model_artifacts/${TIDL_VER}/${SOC} (TIDL_VER comes from VERSION,
#    sourced above; SOC defaults to am62a — same default the launcher uses).
# --------------------------------------------------------------------------
SOC_CHK="${SOC:-am62a}"
ARTIFACTS_DIR="${REPO_DIR}/model_artifacts/${TIDL_VER}/${SOC_CHK}"

if [ ! -d "${REPO_DIR}/models/onnx" ] || [ -z "$(ls -A "${REPO_DIR}/models/onnx" 2>/dev/null)" ]; then
    echo ""
    echo "NOTE: models/onnx/ is empty. Download models before running notebooks:"
    echo "  cd ${REPO_DIR} && ./download_models.sh -y"
fi

if [ ! -d "$ARTIFACTS_DIR" ] || [ -z "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "NOTE: $ARTIFACTS_DIR is empty. The vggish/yamnet (TIDL) demos need compiled"
    echo "      artifacts. Download them with:"
    echo "  cd ${REPO_DIR} && ./download_artifacts.sh -y"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the venv with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Start Jupyter Lab with:"
echo "  ./venv/jupyter_lab_venv.sh"
