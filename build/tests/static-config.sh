#!/usr/bin/env bash

set -Eeuo pipefail

readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly DOCKERFILE="$REPOSITORY_ROOT/build/Dockerfile"
readonly COMPOSE_FILE="$REPOSITORY_ROOT/docker-compose.yaml"
readonly BUILD_WORKFLOW="$REPOSITORY_ROOT/.github/workflows/docker-build.yml"
readonly CLEAR_CACHE_WORKFLOW="$REPOSITORY_ROOT/.github/workflows/clear-cache.yml"
readonly DELETE_OLD_WORKFLOW="$REPOSITORY_ROOT/.github/workflows/delete-old-images.yml"
readonly DELETE_UNTAGGED_WORKFLOW="$REPOSITORY_ROOT/.github/workflows/delete-untagged-images.yml"
readonly BUILD_SCRIPTS="$REPOSITORY_ROOT/build/COPY_ROOT_0/opt/ai-dock/bin/build/layer0"
readonly IMAGE_TEST="$REPOSITORY_ROOT/build/COPY_ROOT_0/opt/ai-dock/bin/tests/verify-pytorch.py"
readonly GPU_VERIFIER="$REPOSITORY_ROOT/build/COPY_ROOT_0/opt/ai-dock/bin/tests/verify-gpu.py"
readonly EXPECTED_BASE="ghcr.io/gabszap/python:3.12-v3-cuda-13.0.3-cudnn-runtime-24.04"
readonly EXPECTED_TAG="2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04"
readonly EXPECTED_INDEX="https://download.pytorch.org/whl/cu130"
readonly EXPECTED_XFORMERS_WHEEL="xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
readonly EXPECTED_XFORMERS_HASH="962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d"

fail() {
    printf "FAIL: %s\n" "$*" >&2
    exit 1
}

assert_contains() {
    local file="$1"
    local expected="$2"
    grep -Fq -- "$expected" "$file" || fail "$file does not contain: $expected"
}

assert_not_contains() {
    local file="$1"
    local forbidden_pattern="$2"
    if grep -Eqi -- "$forbidden_pattern" "$file"; then
        fail "$file contains forbidden pattern: $forbidden_pattern"
    fi
}

assert_contains "$DOCKERFILE" "ARG IMAGE_BASE=\"$EXPECTED_BASE\""
assert_contains "$DOCKERFILE" 'ARG PYTHON_VERSION=3.12'
assert_contains "$DOCKERFILE" 'ARG PYTHON_VENV_NAME=python_312'
assert_contains "$DOCKERFILE" 'ARG PYTORCH_VERSION=2.13.0+cu130'
assert_contains "$DOCKERFILE" 'ARG TORCHVISION_VERSION=0.28.0+cu130'
assert_contains "$DOCKERFILE" 'ARG TORCHAUDIO_VERSION=2.11.0+cu130'
assert_contains "$DOCKERFILE" "ARG PYTORCH_INDEX_URL=$EXPECTED_INDEX"
assert_contains "$DOCKERFILE" 'ARG XFORMERS_VERSION=0.0.35'
assert_contains "$DOCKERFILE" "ARG XFORMERS_INDEX_URL=$EXPECTED_INDEX"
assert_contains "$DOCKERFILE" "ARG XFORMERS_WHEEL=$EXPECTED_XFORMERS_WHEEL"
assert_contains "$DOCKERFILE" "ARG XFORMERS_WHEEL_SHA256=$EXPECTED_XFORMERS_HASH"
assert_contains "$DOCKERFILE" 'org.opencontainers.image.source="https://github.com/gabszap/pytorch"'
assert_contains "$DOCKERFILE" "org.opencontainers.image.version=\"$EXPECTED_TAG\""
assert_contains "$DOCKERFILE" "org.opencontainers.image.base.name=\"$EXPECTED_BASE\""
assert_contains "$DOCKERFILE" 'derived from AI-Dock'

assert_contains "$COMPOSE_FILE" 'platform: linux/amd64'
assert_contains "$COMPOSE_FILE" "IMAGE_BASE: \${IMAGE_BASE:-$EXPECTED_BASE}"
assert_contains "$COMPOSE_FILE" 'PYTHON_VERSION: ${PYTHON_VERSION:-3.12}'
assert_contains "$COMPOSE_FILE" 'PYTHON_VENV_NAME: ${PYTHON_VENV_NAME:-python_312}'
assert_contains "$COMPOSE_FILE" 'PYTORCH_VERSION: ${PYTORCH_VERSION:-2.13.0+cu130}'
assert_contains "$COMPOSE_FILE" 'TORCHVISION_VERSION: ${TORCHVISION_VERSION:-0.28.0+cu130}'
assert_contains "$COMPOSE_FILE" 'TORCHAUDIO_VERSION: ${TORCHAUDIO_VERSION:-2.11.0+cu130}'
assert_contains "$COMPOSE_FILE" "PYTORCH_INDEX_URL: \${PYTORCH_INDEX_URL:-$EXPECTED_INDEX}"
assert_contains "$COMPOSE_FILE" 'XFORMERS_VERSION: ${XFORMERS_VERSION:-0.0.35}'
assert_contains "$COMPOSE_FILE" "XFORMERS_INDEX_URL: \${XFORMERS_INDEX_URL:-$EXPECTED_INDEX}"
assert_contains "$COMPOSE_FILE" "XFORMERS_WHEEL: \${XFORMERS_WHEEL:-$EXPECTED_XFORMERS_WHEEL}"
assert_contains "$COMPOSE_FILE" "XFORMERS_WHEEL_SHA256: \${XFORMERS_WHEEL_SHA256:-$EXPECTED_XFORMERS_HASH}"
assert_contains "$COMPOSE_FILE" "ghcr.io/gabszap/pytorch:\${IMAGE_TAG:-$EXPECTED_TAG}"
assert_contains "$COMPOSE_FILE" 'JUPYTER_TYPE=${JUPYTER_TYPE:-lab}'
assert_not_contains "$COMPOSE_FILE" '/dev/kfd|rocm|docker\.io|dockerhub'

assert_contains "$BUILD_WORKFLOW" 'workflow_dispatch:'
assert_contains "$BUILD_WORKFLOW" 'contents: read'
assert_contains "$BUILD_WORKFLOW" 'packages: write'
assert_contains "$BUILD_WORKFLOW" "IMAGE_BASE: $EXPECTED_BASE"
assert_contains "$BUILD_WORKFLOW" "IMAGE_VERSION: $EXPECTED_TAG"
assert_contains "$BUILD_WORKFLOW" 'PYTORCH_VERSION: 2.13.0+cu130'
assert_contains "$BUILD_WORKFLOW" 'TORCHVISION_VERSION: 0.28.0+cu130'
assert_contains "$BUILD_WORKFLOW" 'TORCHAUDIO_VERSION: 2.11.0+cu130'
assert_contains "$BUILD_WORKFLOW" "PYTORCH_INDEX_URL: $EXPECTED_INDEX"
assert_contains "$BUILD_WORKFLOW" 'XFORMERS_VERSION: 0.0.35'
assert_contains "$BUILD_WORKFLOW" "XFORMERS_INDEX_URL: $EXPECTED_INDEX"
assert_contains "$BUILD_WORKFLOW" "XFORMERS_WHEEL: $EXPECTED_XFORMERS_WHEEL"
assert_contains "$BUILD_WORKFLOW" "XFORMERS_WHEEL_SHA256: $EXPECTED_XFORMERS_HASH"
assert_contains "$BUILD_WORKFLOW" 'uses: actions/checkout@v4'
assert_contains "$BUILD_WORKFLOW" 'uses: docker/login-action@v3'
assert_contains "$BUILD_WORKFLOW" 'uses: docker/build-push-action@v6'
assert_contains "$BUILD_WORKFLOW" 'IMAGE_NAME=ghcr.io/${GITHUB_REPOSITORY,,}'
assert_contains "$BUILD_WORKFLOW" '${{ env.IMAGE_NAME }}:${{ env.IMAGE_VERSION }}'
assert_contains "$BUILD_WORKFLOW" '${{ env.IMAGE_NAME }}:latest-cuda'
assert_contains "$BUILD_WORKFLOW" '${{ env.IMAGE_NAME }}:latest'
assert_contains "$BUILD_WORKFLOW" 'provenance: false'

if [[ "$(grep -Fc 'uses: docker/build-push-action@v6' "$BUILD_WORKFLOW")" -ne 1 ]]; then
    fail "the build workflow must contain exactly one image build"
fi
assert_not_contains "$BUILD_WORKFLOW" '^[[:space:]]{2}push:|pull_request:|matrix:|dockerhub|docker\.io|rocm|amd|cpu|cuda-1[12]|v2-'

assert_contains "$CLEAR_CACHE_WORKFLOW" 'uses: actions/github-script@v7'
for cleanup_workflow in "$DELETE_OLD_WORKFLOW" "$DELETE_UNTAGGED_WORKFLOW"; do
    assert_contains "$cleanup_workflow" 'contents: read'
    assert_contains "$cleanup_workflow" 'packages: write'
    assert_contains "$cleanup_workflow" 'uses: actions/github-script@v7'
    assert_contains "$cleanup_workflow" 'github-token: ${{ secrets.GITHUB_TOKEN }}'
    assert_contains "$cleanup_workflow" 'github.paginate'
    assert_contains "$cleanup_workflow" '/users/${encodedOwner}/packages/container/${encodedPackageName}/versions'
    assert_contains "$cleanup_workflow" 'group: package-cleanup-${{ github.repository_owner }}-${{ github.event.repository.name }}'
    assert_not_contains "$cleanup_workflow" 'DELETE_PACKAGES_TOKEN|orgs/|github-script@v6'
done
assert_contains "$DELETE_OLD_WORKFLOW" 'new Set(["latest", "latest-cuda"])'
assert_contains "$DELETE_UNTAGGED_WORKFLOW" "workflow_run.conclusion == 'success'"

[[ ! -e "$BUILD_SCRIPTS/amd.sh" ]] || fail "obsolete AMD build script still exists"
[[ ! -e "$BUILD_SCRIPTS/cpu.sh" ]] || fail "obsolete CPU build script still exists"
assert_contains "$BUILD_SCRIPTS/init.sh" 'this image only supports NVIDIA_GPU'
assert_contains "$BUILD_SCRIPTS/common.sh" 'this image only supports 3.12'
assert_contains "$BUILD_SCRIPTS/common.sh" 'expected python_312'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '--index-url "$PYTORCH_INDEX_URL"'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '"torch==$PYTORCH_VERSION"'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '"torchvision==$TORCHVISION_VERSION"'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '"torchaudio==$TORCHAUDIO_VERSION"'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '--index-url "$XFORMERS_INDEX_URL"'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '--no-deps'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '--require-hashes'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '--report "$XFORMERS_INSTALL_REPORT"'
assert_contains "$BUILD_SCRIPTS/nvidia.sh" '"$PYTHON_VENV_PYTHON" -m xformers.info'
assert_not_contains "$BUILD_SCRIPTS/nvidia.sh" 'extra-index-url|flash.attn|sageattention'

assert_contains "$IMAGE_TEST" 'EXPECTED_XFORMERS_VERSION = "0.0.35"'
assert_contains "$IMAGE_TEST" "EXPECTED_XFORMERS_SHA256 = \"$EXPECTED_XFORMERS_HASH\""
assert_contains "$IMAGE_TEST" 'xformers-install-report.json'
assert_contains "$IMAGE_TEST" 'xformers-torch-versions.json'
assert_contains "$IMAGE_TEST" 'torch>=2.10'
assert_contains "$IMAGE_TEST" 'memory_efficient_attention'
assert_contains "$IMAGE_TEST" '"flash-attn-3"'
assert_contains "$IMAGE_TEST" '"flash-attn-4"'
assert_contains "$IMAGE_TEST" '"sageattention"'
assert_contains "$IMAGE_TEST" 'ERROR: No CUDA GPU is available.'

[[ -f "$GPU_VERIFIER" ]] || fail "GPU verifier is missing"
[[ -x "$GPU_VERIFIER" ]] || fail "GPU verifier is not executable"
assert_contains "$GPU_VERIFIER" '#!/opt/environments/python/python_312/bin/python'
assert_contains "$GPU_VERIFIER" 'torch.float16, torch.bfloat16'
assert_contains "$GPU_VERIFIER" 'scaled_dot_product_attention('
assert_contains "$GPU_VERIFIER" 'xformers_ops.memory_efficient_attention('
assert_contains "$GPU_VERIFIER" 'layout=BMHK'
assert_contains "$GPU_VERIFIER" '"-m", "xformers.info"'

readonly GPU_WRAPPER="$REPOSITORY_ROOT/build/tests/gpu-validation.sh"
assert_contains "$GPU_WRAPPER" '/opt/ai-dock/bin/tests/verify-gpu.py'
assert_contains "$GPU_WRAPPER" '--expected-capability "$EXPECTED_CAPABILITY_INPUT"'
assert_not_contains "$GPU_WRAPPER" 'memory_efficient_attention|scaled_dot_product_attention'

python3 - "$IMAGE_TEST" "$GPU_VERIFIER" "$REPOSITORY_ROOT/build/tests/cleanup-workflows.py" <<'PY'
import ast
import pathlib
import runpy
import sys

for source_path in sys.argv[1:]:
    ast.parse(pathlib.Path(source_path).read_text(encoding="utf-8"), filename=source_path)

gpu_verifier = runpy.run_path(sys.argv[2], run_name="verify_gpu_import_check")
if not callable(gpu_verifier.get("main")):
    raise SystemExit("GPU verifier import did not define main")
PY
python3 "$GPU_VERIFIER" --help >/dev/null
python3 "$REPOSITORY_ROOT/build/tests/cleanup-workflows.py"

printf "Static configuration checks passed.\n"
