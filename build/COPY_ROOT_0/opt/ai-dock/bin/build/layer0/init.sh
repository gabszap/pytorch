#!/bin/bash

# Must exit and fail to build if any command fails
set -Eeuo pipefail
umask 002

source /opt/ai-dock/etc/environment.sh

if [[ "${XPU_TARGET:-}" != "NVIDIA_GPU" ]]; then
    printf "Unsupported XPU_TARGET %q; this image only supports NVIDIA_GPU.\n" "${XPU_TARGET:-<unset>}" >&2
    exit 1
fi

PYTHON_VENV="$VENV_DIR/$PYTHON_DEFAULT_VENV"
PYTHON_VENV_PYTHON="$PYTHON_VENV/bin/python"
PYTHON_VENV_PIP="$PYTHON_VENV/bin/pip"

source /opt/ai-dock/bin/build/layer0/common.sh
source /opt/ai-dock/bin/build/layer0/nvidia.sh
source /opt/ai-dock/bin/build/layer0/clean.sh
