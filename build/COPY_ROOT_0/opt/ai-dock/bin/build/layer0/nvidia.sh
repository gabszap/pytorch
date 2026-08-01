#!/bin/false

build_nvidia_main() {
    build_nvidia_validate_configuration
    build_nvidia_install_torch
    build_nvidia_install_xformers
    build_nvidia_record_xformers_info
    build_nvidia_run_tests
}

build_nvidia_validate_configuration() {
    [[ "${CUDA_VERSION:-}" == 13.0.* ]] || {
        printf "Unsupported base CUDA version %q; expected 13.0.x.\n" "${CUDA_VERSION:-<unset>}" >&2
        exit 1
    }
    [[ "${PYTORCH_VERSION:-}" == "2.13.0+cu130" ]] || {
        printf "Unsupported PYTORCH_VERSION %q; expected 2.13.0+cu130.\n" "${PYTORCH_VERSION:-<unset>}" >&2
        exit 1
    }
    [[ "${TORCHVISION_VERSION:-}" == "0.28.0+cu130" ]] || {
        printf "Unsupported TORCHVISION_VERSION %q; expected 0.28.0+cu130.\n" "${TORCHVISION_VERSION:-<unset>}" >&2
        exit 1
    }
    [[ "${TORCHAUDIO_VERSION:-}" == "2.11.0+cu130" ]] || {
        printf "Unsupported TORCHAUDIO_VERSION %q; expected 2.11.0+cu130.\n" "${TORCHAUDIO_VERSION:-<unset>}" >&2
        exit 1
    }
    [[ "${PYTORCH_INDEX_URL:-}" == "https://download.pytorch.org/whl/cu130" ]] || {
        printf "Unsupported PYTORCH_INDEX_URL %q; expected https://download.pytorch.org/whl/cu130.\n" "${PYTORCH_INDEX_URL:-<unset>}" >&2
        exit 1
    }
    [[ "${XFORMERS_VERSION:-}" == "0.0.35" ]] || {
        printf "Unsupported XFORMERS_VERSION %q; expected 0.0.35.\n" "${XFORMERS_VERSION:-<unset>}" >&2
        exit 1
    }
    [[ "${XFORMERS_INDEX_URL:-}" == "https://download.pytorch.org/whl/cu130" ]] || {
        printf "Unsupported XFORMERS_INDEX_URL %q; expected https://download.pytorch.org/whl/cu130.\n" "${XFORMERS_INDEX_URL:-<unset>}" >&2
        exit 1
    }
    [[ "${XFORMERS_WHEEL:-}" == "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl" ]] || {
        printf "Unsupported XFORMERS_WHEEL %q; expected xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl.\n" "${XFORMERS_WHEEL:-<unset>}" >&2
        exit 1
    }
    [[ "${XFORMERS_WHEEL_SHA256:-}" == "962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d" ]] || {
        printf "Unsupported XFORMERS_WHEEL_SHA256 %q; expected the pinned official wheel hash.\n" "${XFORMERS_WHEEL_SHA256:-<unset>}" >&2
        exit 1
    }
}

build_nvidia_install_torch() {
    install -d -o root -g root -m 0755 "$(dirname "$PYTORCH_INSTALL_REPORT")"
    "$PYTHON_VENV_PIP" install --no-cache-dir \
        --index-url "$PYTORCH_INDEX_URL" \
        --report "$PYTORCH_INSTALL_REPORT" \
        "torch==$PYTORCH_VERSION" \
        "torchvision==$TORCHVISION_VERSION" \
        "torchaudio==$TORCHAUDIO_VERSION"
}

build_nvidia_read_protected_versions() {
    "$PYTHON_VENV_PYTHON" -c 'import json; from importlib import metadata; print(json.dumps({name: metadata.version(name) for name in ("torch", "torchvision", "torchaudio", "numpy")}, sort_keys=True))'
}

build_nvidia_install_xformers() {
    local requirements_file
    local torch_versions_before
    local torch_versions_after

    torch_versions_before=$(build_nvidia_read_protected_versions)
    requirements_file=$(mktemp)
    printf 'xformers==%s --hash=sha256:%s\n' "$XFORMERS_VERSION" "$XFORMERS_WHEEL_SHA256" >"$requirements_file"

    "$PYTHON_VENV_PIP" install --no-cache-dir \
        --index-url "$XFORMERS_INDEX_URL" \
        --no-deps \
        --require-hashes \
        --report "$XFORMERS_INSTALL_REPORT" \
        --requirement "$requirements_file"
    rm -f "$requirements_file"

    torch_versions_after=$(build_nvidia_read_protected_versions)
    if [[ "$torch_versions_after" != "$torch_versions_before" ]]; then
        printf "Installing xformers changed the torch stack: before=%s after=%s.\n" \
            "$torch_versions_before" "$torch_versions_after" >&2
        exit 1
    fi
    printf '{"before":%s,"after":%s}\n' \
        "$torch_versions_before" "$torch_versions_after" >"$XFORMERS_TORCH_SNAPSHOT"
}

build_nvidia_record_xformers_info() {
    {
        printf 'Build-time xformers metadata only; CUDA kernels were not executed.\n'
        "$PYTHON_VENV_PYTHON" -m xformers.info
    } 2>&1 | tee "$XFORMERS_INFO_REPORT"
}

build_nvidia_run_tests() {
    "$PYTHON_VENV_PIP" check
    "$PYTHON_VENV_PYTHON" /opt/ai-dock/bin/tests/verify-pytorch.py
}

build_nvidia_main "$@"
