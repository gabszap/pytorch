#!/bin/false

readonly PYTORCH_INSTALL_REPORT="/opt/ai-dock/share/pytorch-install-report.json"
readonly XFORMERS_INSTALL_REPORT="/opt/ai-dock/share/xformers-install-report.json"
readonly XFORMERS_INFO_REPORT="/opt/ai-dock/share/xformers-info.txt"
readonly XFORMERS_TORCH_SNAPSHOT="/opt/ai-dock/share/xformers-torch-versions.json"

build_common_main() {
    build_common_validate_environment
}

build_common_validate_environment() {
    if [[ "${PYTHON_VERSION:-}" != "3.12" ]]; then
        printf "Unsupported PYTHON_VERSION %q; this image only supports 3.12.\n" "${PYTHON_VERSION:-<unset>}" >&2
        exit 1
    fi
    if [[ "${PYTHON_DEFAULT_VENV:-}" != "python_312" ]]; then
        printf "Unsupported default venv %q; expected python_312.\n" "${PYTHON_DEFAULT_VENV:-<unset>}" >&2
        exit 1
    fi
    if [[ "${PYTHON_VENV_NAME:-}" != "python_312" ]]; then
        printf "Unsupported PYTHON_VENV_NAME %q; expected python_312.\n" "${PYTHON_VENV_NAME:-<unset>}" >&2
        exit 1
    fi
    if [[ ! -x "$PYTHON_VENV_PYTHON" || ! -x "$PYTHON_VENV_PIP" ]]; then
        printf "The inherited python_312 virtual environment is incomplete at %s.\n" "$PYTHON_VENV" >&2
        exit 1
    fi

    local installed_python_version
    installed_python_version=$("$PYTHON_VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$installed_python_version" != "3.12" ]]; then
        printf "Expected Python 3.12 in %s but found %s.\n" "$PYTHON_VENV" "$installed_python_version" >&2
        exit 1
    fi
}

build_common_main "$@"
