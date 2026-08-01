#!/usr/bin/env bash

set -Eeuo pipefail

readonly IMAGE_NAME="${1:?Usage: runtime-image.sh IMAGE_NAME}"
readonly SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly EXPECTED_BASE="ghcr.io/gabszap/python:3.12-v3-cuda-13.0.3-cudnn-runtime-24.04"
readonly EXPECTED_VERSION="2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04"

fail() {
    printf "FAIL: %s\n" "$*" >&2
    exit 1
}

docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || fail "image does not exist locally: $IMAGE_NAME"

docker image inspect "$IMAGE_NAME" --format '{{json .Config.Labels}}' | python3 -c '
import json
import sys

labels = json.load(sys.stdin)
expected = {
    "org.opencontainers.image.source": "https://github.com/gabszap/pytorch",
    "org.opencontainers.image.version": "'"$EXPECTED_VERSION"'",
    "org.opencontainers.image.base.name": "'"$EXPECTED_BASE"'",
    "org.opencontainers.image.licenses": "LicenseRef-AI-Dock-Custom",
    "maintainer": "gabszap",
}
for name, value in expected.items():
    if labels.get(name) != value:
        raise SystemExit(f"unexpected label {name}: {labels.get(name)!r}")
'

docker run --rm \
    --entrypoint /bin/bash \
    --volume "$SCRIPT_DIRECTORY/runtime-container.sh:/tmp/runtime-container.sh:ro" \
    "$IMAGE_NAME" \
    /tmp/runtime-container.sh

printf "Runtime image checks passed for %s.\n" "$IMAGE_NAME"
