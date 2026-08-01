#!/usr/bin/env bash

set -Eeuo pipefail

readonly PYTHON_VENV="/opt/environments/python/python_312"
readonly JUPYTER_VENV="/opt/environments/python/jupyter"
readonly KERNEL_JSON="/usr/local/share/jupyter/kernels/python_312/kernel.json"
readonly GPU_VERIFIER="/opt/ai-dock/bin/tests/verify-gpu.py"

fail() {
    printf "FAIL: %s\n" "$*" >&2
    exit 1
}

stop_jupyter() {
    if [[ -z "${jupyter_pid:-}" ]]; then
        return
    fi
    kill "$jupyter_pid" >/dev/null 2>&1 || true
    wait "$jupyter_pid" >/dev/null 2>&1 || true
    jupyter_pid=""
}

source /opt/ai-dock/etc/environment.sh
source /etc/os-release
[[ "$VERSION_ID" == "24.04" ]] || fail "expected Ubuntu 24.04, found $VERSION_ID"
[[ "${XPU_TARGET:-}" == "NVIDIA_GPU" ]] || fail "expected NVIDIA_GPU target"
[[ "${CUDA_VERSION:-}" == 13.0.* ]] || fail "expected base CUDA 13.0.x"
[[ -x "$PYTHON_VENV/bin/python" ]] || fail "python_312 virtual environment is missing"
[[ -x "$JUPYTER_VENV/bin/jupyter" ]] || fail "Jupyter virtual environment is missing"
[[ -f "$GPU_VERIFIER" ]] || fail "GPU verifier is missing"
[[ -x "$GPU_VERIFIER" ]] || fail "GPU verifier is not executable"

"$PYTHON_VENV/bin/pip" check
"$PYTHON_VENV/bin/python" /opt/ai-dock/bin/tests/verify-pytorch.py
"$PYTHON_VENV/bin/python" "$GPU_VERIFIER" --help | grep -Fq -- '--expected-capability'
if gpu_verifier_output=$(CUDA_VISIBLE_DEVICES="" "$PYTHON_VENV/bin/python" "$GPU_VERIFIER" 2>&1); then
    fail "GPU verifier unexpectedly passed while CUDA devices were hidden"
fi
[[ "$gpu_verifier_output" == *"ERROR: No CUDA GPU is available."* ]] || {
    fail "GPU verifier did not report its explicit no-GPU failure: $gpu_verifier_output"
}
"$JUPYTER_VENV/bin/python" -c 'import jupyterlab, notebook'

[[ -f "$KERNEL_JSON" ]] || fail "python_312 kernel is not registered"
"$JUPYTER_VENV/bin/python" - "$KERNEL_JSON" "$PYTHON_VENV/bin/python" <<'PY'
import json
import pathlib
import sys

kernel = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if kernel["argv"][0] != sys.argv[2]:
    raise SystemExit(f"unexpected kernel interpreter: {kernel['argv'][0]}")
PY
"$JUPYTER_VENV/bin/jupyter" kernelspec list --json | grep -Fq 'python_312'

test_home="$(mktemp -d)"
test_root="$(mktemp -d)"
response_file="$(mktemp)"
test_port=19888
trap 'stop_jupyter; rm -rf "$test_home" "$test_root" "$response_file"' EXIT

HOME="$test_home" "$JUPYTER_VENV/bin/jupyter" lab \
    --allow-root \
    --ip=127.0.0.1 \
    --port="$test_port" \
    --no-browser \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.root_dir="$test_root" \
    >"$test_root/jupyter-test.log" 2>&1 &
jupyter_pid=$!

for _ in {1..60}; do
    if curl --fail --silent "http://127.0.0.1:$test_port/api/status" >"$response_file"; then
        break
    fi
    if ! kill -0 "$jupyter_pid" >/dev/null 2>&1; then
        cat "$test_root/jupyter-test.log" >&2
        fail "Jupyter exited before becoming healthy"
    fi
    sleep 1
done
curl --fail --silent --show-error "http://127.0.0.1:$test_port/api/status" >"$response_file"
"$JUPYTER_VENV/bin/python" - "$response_file" <<'PY'
import json
import pathlib
import sys

status = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if "connections" not in status or "kernels" not in status:
    raise SystemExit(f"unexpected Jupyter status: {status!r}")
PY

stop_jupyter
printf "Container runtime, PyTorch, xformers import/info, kernel, and Jupyter checks passed; CUDA kernels were not tested.\n"
