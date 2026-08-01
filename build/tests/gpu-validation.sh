#!/usr/bin/env bash

set -Eeuo pipefail

readonly IMAGE_NAME="${1:?Usage: gpu-validation.sh IMAGE_NAME [EXPECTED_CAPABILITY]}"
readonly EXPECTED_CAPABILITY_INPUT="${2:-${EXPECTED_CAPABILITY:-}}"
readonly PYTHON_BIN="/opt/environments/python/python_312/bin/python"
readonly GPU_VERIFIER="/opt/ai-dock/bin/tests/verify-gpu.py"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    printf "SKIP: no NVIDIA GPU is exposed on the host; GPU validation was not run.\n"
    exit 0
fi

verifier_arguments=()
if [[ -n "$EXPECTED_CAPABILITY_INPUT" ]]; then
    verifier_arguments+=(--expected-capability "$EXPECTED_CAPABILITY_INPUT")
fi

docker run --rm --gpus all \
    --entrypoint "$PYTHON_BIN" \
    "$IMAGE_NAME" \
    "$GPU_VERIFIER" "${verifier_arguments[@]}"
