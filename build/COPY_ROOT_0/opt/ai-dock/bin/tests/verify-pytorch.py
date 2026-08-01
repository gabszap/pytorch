#!/usr/bin/env python3

import ast
import ctypes
import importlib.util
import json
import os
import re
import runpy
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


EXPECTED_TORCH_DISTRIBUTIONS = {
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "torchaudio": "2.11.0+cu130",
}
EXPECTED_XFORMERS_VERSION = "0.0.35"
EXPECTED_XFORMERS_WHEEL = "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
EXPECTED_XFORMERS_SHA256 = "962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d"
EXPECTED_WHEEL_SHA256 = {
    "torch": "8db7338e6895c3d4bd89a02ff4209507d1f0cf2ffeb3b898538b5a07d1ea8c1e",
    "torchvision": "8a0008d34ccc4e81066b97ff0ae5a34c676bfdf3464baf40c01b320dc9a45ce0",
    "torchaudio": "3fba988f4301fe13547fe5e99c76d9ae36a27e19ded82eeffed9d2456e12edef",
}
EXPECTED_INDEX_URL = "https://download.pytorch.org/whl/cu130"
EXPECTED_XFORMERS_URL = f"{EXPECTED_INDEX_URL}/{EXPECTED_XFORMERS_WHEEL}"
INSTALL_REPORT = Path("/opt/ai-dock/share/pytorch-install-report.json")
XFORMERS_INSTALL_REPORT = Path("/opt/ai-dock/share/xformers-install-report.json")
XFORMERS_INFO_REPORT = Path("/opt/ai-dock/share/xformers-info.txt")
XFORMERS_TORCH_SNAPSHOT = Path("/opt/ai-dock/share/xformers-torch-versions.json")
GPU_VERIFIER = Path("/opt/ai-dock/bin/tests/verify-gpu.py")
FORBIDDEN_DISTRIBUTIONS = {
    "flash-attn",
    "flash-attn-3",
    "flash-attn-4",
    "sageattention",
}
FORBIDDEN_MODULES = ("flash_attn", "flash_attn_3", "flash_attn_4", "sageattention")


def require_environment_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is missing.")
    return value


def normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def assert_exact_environment() -> None:
    expected_environment = {
        "XPU_TARGET": "NVIDIA_GPU",
        "PYTHON_VERSION": "3.12",
        "PYTHON_VENV_NAME": "python_312",
        "PYTHON_DEFAULT_VENV": "python_312",
        "PYTORCH_VERSION": EXPECTED_TORCH_DISTRIBUTIONS["torch"],
        "TORCHVISION_VERSION": EXPECTED_TORCH_DISTRIBUTIONS["torchvision"],
        "TORCHAUDIO_VERSION": EXPECTED_TORCH_DISTRIBUTIONS["torchaudio"],
        "PYTORCH_INDEX_URL": EXPECTED_INDEX_URL,
        "XFORMERS_VERSION": EXPECTED_XFORMERS_VERSION,
        "XFORMERS_INDEX_URL": EXPECTED_INDEX_URL,
        "XFORMERS_WHEEL": EXPECTED_XFORMERS_WHEEL,
        "XFORMERS_WHEEL_SHA256": EXPECTED_XFORMERS_SHA256,
    }
    for name, expected_value in expected_environment.items():
        actual_value = require_environment_value(name)
        if actual_value != expected_value:
            raise RuntimeError(f"Expected {name}={expected_value!r}, found {actual_value!r}.")

    base_cuda_version = require_environment_value("CUDA_VERSION")
    if not base_cuda_version.startswith("13.0."):
        raise RuntimeError(f"Expected base CUDA 13.0.x, found {base_cuda_version!r}.")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Expected Python 3.12, found {sys.version.split()[0]}.")


def assert_distribution_metadata() -> None:
    for distribution_name, expected_version in EXPECTED_TORCH_DISTRIBUTIONS.items():
        installed_distribution = metadata.distribution(distribution_name)
        if installed_distribution.version != expected_version:
            raise RuntimeError(
                f"Expected {distribution_name} {expected_version}, "
                f"found {installed_distribution.version}."
            )

        wheel_metadata = installed_distribution.read_text("WHEEL") or ""
        wheel_tags = re.findall(r"^Tag: (.+)$", wheel_metadata, flags=re.MULTILINE)
        expected_wheel_tags = {
            "cp312-cp312-linux_x86_64",
            "cp312-cp312-manylinux_2_28_x86_64",
        }
        if expected_wheel_tags.isdisjoint(wheel_tags):
            raise RuntimeError(
                f"{distribution_name} does not contain the expected cp312 Linux amd64 "
                f"wheel tag; found {wheel_tags}."
            )

    xformers_distribution = metadata.distribution("xformers")
    if xformers_distribution.version != EXPECTED_XFORMERS_VERSION:
        raise RuntimeError(
            f"Expected xformers {EXPECTED_XFORMERS_VERSION}, "
            f"found {xformers_distribution.version}."
        )
    xformers_wheel_metadata = xformers_distribution.read_text("WHEEL") or ""
    xformers_wheel_tags = re.findall(
        r"^Tag: (.+)$", xformers_wheel_metadata, flags=re.MULTILINE
    )
    expected_xformers_tag = "py39-none-manylinux_2_28_x86_64"
    if xformers_wheel_tags != [expected_xformers_tag]:
        raise RuntimeError(
            f"Expected xformers wheel tag {expected_xformers_tag!r}, "
            f"found {xformers_wheel_tags}."
        )

    xformers_metadata = xformers_distribution.metadata
    if xformers_metadata.get("Requires-Python") != ">=3.9":
        raise RuntimeError(
            "Expected xformers stable Python ABI metadata Requires-Python >=3.9, "
            f"found {xformers_metadata.get('Requires-Python')!r}."
        )
    xformers_requirements = xformers_metadata.get_all("Requires-Dist") or []
    normalized_requirements = {requirement.replace(" ", "") for requirement in xformers_requirements}
    if "torch>=2.10" not in normalized_requirements:
        raise RuntimeError(
            "Expected xformers metadata to declare torch>=2.10; "
            f"found {xformers_requirements}."
        )

    installed_names = {
        normalized_distribution_name(distribution.metadata["Name"])
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }
    unexpected_distributions = sorted(FORBIDDEN_DISTRIBUTIONS & installed_names)
    if unexpected_distributions:
        raise RuntimeError(
            f"Forbidden optional attention distributions are installed: {unexpected_distributions}."
        )

    unexpected_modules = [
        module_name
        for module_name in FORBIDDEN_MODULES
        if importlib.util.find_spec(module_name) is not None
    ]
    if unexpected_modules:
        raise RuntimeError(f"Forbidden optional attention modules are importable: {unexpected_modules}.")


def assert_install_report() -> None:
    if not INSTALL_REPORT.is_file():
        raise RuntimeError(f"PyTorch pip install report is missing at {INSTALL_REPORT}.")

    report = json.loads(INSTALL_REPORT.read_text(encoding="utf-8"))
    requested_downloads = {}
    for item in report.get("install", []):
        package_metadata = item.get("metadata", {})
        normalized_name = normalized_distribution_name(package_metadata.get("name", ""))
        if normalized_name not in EXPECTED_TORCH_DISTRIBUTIONS or not item.get("requested"):
            continue
        requested_downloads[normalized_name] = item.get("download_info", {})

    if set(requested_downloads) != set(EXPECTED_TORCH_DISTRIBUTIONS):
        raise RuntimeError(
            "Install report does not identify all three requested PyTorch distributions: "
            f"{sorted(requested_downloads)}."
        )

    for distribution_name, download_info in requested_downloads.items():
        download_url = download_info.get("url", "")
        parsed_url = urlparse(download_url)
        expected_wheel_prefix = (
            f"{distribution_name}-{EXPECTED_TORCH_DISTRIBUTIONS[distribution_name]}-cp312-cp312-"
        )
        wheel_name = unquote(Path(parsed_url.path).name)
        if parsed_url.scheme != "https" or parsed_url.hostname not in {
            "download.pytorch.org",
            "download-r2.pytorch.org",
        }:
            raise RuntimeError(f"Unexpected download host for {distribution_name}: {download_url}.")
        if "/whl/cu130/" not in parsed_url.path or not wheel_name.startswith(expected_wheel_prefix):
            raise RuntimeError(f"Unexpected wheel URL for {distribution_name}: {download_url}.")

        archive_hash = download_info.get("archive_info", {}).get("hashes", {}).get("sha256")
        expected_hash = EXPECTED_WHEEL_SHA256[distribution_name]
        if archive_hash != expected_hash:
            raise RuntimeError(
                f"Unexpected SHA-256 for {distribution_name}: {archive_hash!r}; "
                f"expected {expected_hash}."
            )


def assert_xformers_install_report() -> None:
    if not XFORMERS_INSTALL_REPORT.is_file():
        raise RuntimeError(
            f"xformers pip install report is missing at {XFORMERS_INSTALL_REPORT}."
        )

    report = json.loads(XFORMERS_INSTALL_REPORT.read_text(encoding="utf-8"))
    report_environment = report.get("environment", {})
    expected_report_environment = {
        "platform_machine": "x86_64",
        "python_version": "3.12",
        "sys_platform": "linux",
    }
    for name, expected_value in expected_report_environment.items():
        if report_environment.get(name) != expected_value:
            raise RuntimeError(
                f"Expected xformers report environment {name}={expected_value!r}, "
                f"found {report_environment.get(name)!r}."
            )

    installed_items = report.get("install", [])
    if len(installed_items) != 1:
        raise RuntimeError(
            "The --no-deps xformers report must contain exactly one install item; "
            f"found {len(installed_items)}."
        )

    item = installed_items[0]
    package_metadata = item.get("metadata", {})
    if normalized_distribution_name(package_metadata.get("name", "")) != "xformers":
        raise RuntimeError(f"Unexpected distribution in xformers report: {package_metadata!r}.")
    if package_metadata.get("version") != EXPECTED_XFORMERS_VERSION:
        raise RuntimeError(f"Unexpected xformers report version: {package_metadata!r}.")
    if item.get("requested") is not True:
        raise RuntimeError("The xformers report does not mark xformers as directly requested.")

    download_info = item.get("download_info", {})
    download_url = unquote(download_info.get("url", ""))
    if download_url != EXPECTED_XFORMERS_URL:
        raise RuntimeError(
            f"Unexpected xformers wheel URL {download_url!r}; "
            f"expected {EXPECTED_XFORMERS_URL!r}."
        )
    archive_hash = download_info.get("archive_info", {}).get("hashes", {}).get("sha256")
    if archive_hash != EXPECTED_XFORMERS_SHA256:
        raise RuntimeError(
            f"Unexpected xformers SHA-256 {archive_hash!r}; "
            f"expected {EXPECTED_XFORMERS_SHA256}."
        )


def assert_torch_unchanged_by_xformers() -> None:
    if not XFORMERS_TORCH_SNAPSHOT.is_file():
        raise RuntimeError(
            f"xformers torch-version snapshot is missing at {XFORMERS_TORCH_SNAPSHOT}."
        )

    snapshot = json.loads(XFORMERS_TORCH_SNAPSHOT.read_text(encoding="utf-8"))
    versions_before = snapshot.get("before", {})
    versions_after = snapshot.get("after", {})
    torch_versions_before = {
        name: versions_before.get(name) for name in EXPECTED_TORCH_DISTRIBUTIONS
    }
    if torch_versions_before != EXPECTED_TORCH_DISTRIBUTIONS:
        raise RuntimeError(f"Unexpected torch stack before xformers installation: {snapshot!r}.")
    if not versions_before.get("numpy"):
        raise RuntimeError(f"The pre-xformers NumPy version is missing: {snapshot!r}.")
    if versions_after != versions_before:
        raise RuntimeError(f"xformers changed torch or NumPy versions: {snapshot!r}.")


def assert_xformers_import_and_info() -> None:
    import torch
    import xformers
    import xformers.ops as xformers_ops
    from xformers.info import get_features_status

    if xformers.__version__ != EXPECTED_XFORMERS_VERSION:
        raise RuntimeError(
            f"Expected imported xformers {EXPECTED_XFORMERS_VERSION}, "
            f"found {xformers.__version__}."
        )
    if not callable(xformers_ops.memory_efficient_attention):
        raise RuntimeError("xformers.ops.memory_efficient_attention is not callable.")

    backend_metadata = {
        name: status
        for name, status in get_features_status().items()
        if name.startswith("memory_efficient_attention.")
    }
    if not backend_metadata:
        raise RuntimeError("xformers did not expose registered attention backend metadata.")

    info_result = subprocess.run(
        [sys.executable, "-m", "xformers.info"],
        check=True,
        capture_output=True,
        text=True,
    )
    expected_info_fragments = (
        f"xFormers {EXPECTED_XFORMERS_VERSION}",
        f"pytorch.version:                                   {EXPECTED_TORCH_DISTRIBUTIONS['torch']}",
    )
    for expected_fragment in expected_info_fragments:
        if expected_fragment not in info_result.stdout:
            raise RuntimeError(f"xformers.info output is missing {expected_fragment!r}.")

    if not XFORMERS_INFO_REPORT.is_file():
        raise RuntimeError(f"Build-time xformers info is missing at {XFORMERS_INFO_REPORT}.")
    saved_info = XFORMERS_INFO_REPORT.read_text(encoding="utf-8")
    for expected_fragment in (
        "Build-time xformers metadata only; CUDA kernels were not executed.",
        *expected_info_fragments,
    ):
        if expected_fragment not in saved_info:
            raise RuntimeError(f"Saved xformers info is missing {expected_fragment!r}.")

    backend_summary = ", ".join(
        f"{name}={status}" for name, status in sorted(backend_metadata.items())
    )
    print(f"Registered xformers attention backends: {backend_summary}")
    if not torch.cuda.is_available():
        print(
            "Verified xformers import and backend metadata without a GPU; "
            "CUDA attention kernels were not executed."
        )


def assert_gpu_verifier() -> None:
    if not GPU_VERIFIER.is_file():
        raise RuntimeError(f"GPU verifier is missing at {GPU_VERIFIER}.")
    if not GPU_VERIFIER.stat().st_mode & 0o111:
        raise RuntimeError(f"GPU verifier is not executable: {GPU_VERIFIER}.")

    source = GPU_VERIFIER.read_text(encoding="utf-8")
    ast.parse(source, filename=str(GPU_VERIFIER))
    imported_module = runpy.run_path(str(GPU_VERIFIER), run_name="verify_gpu_import_check")
    if not callable(imported_module.get("main")):
        raise RuntimeError("GPU verifier import did not define a callable main function.")

    help_result = subprocess.run(
        [sys.executable, str(GPU_VERIFIER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for expected_option in ("--expected-capability", "--matmul-size", "--sequence-length"):
        if expected_option not in help_result.stdout:
            raise RuntimeError(f"GPU verifier help is missing {expected_option!r}.")

    no_gpu_environment = os.environ.copy()
    no_gpu_environment["CUDA_VISIBLE_DEVICES"] = ""
    no_gpu_result = subprocess.run(
        [sys.executable, str(GPU_VERIFIER)],
        check=False,
        capture_output=True,
        text=True,
        env=no_gpu_environment,
    )
    combined_output = no_gpu_result.stdout + no_gpu_result.stderr
    if no_gpu_result.returncode == 0:
        raise RuntimeError("GPU verifier unexpectedly passed while CUDA devices were hidden.")
    if "ERROR: No CUDA GPU is available." not in combined_output:
        raise RuntimeError(
            "GPU verifier did not report its explicit no-GPU failure; "
            f"output was {combined_output!r}."
        )


def assert_cuda_and_cudnn() -> None:
    import torch
    import torchaudio
    import torchvision

    del torchaudio, torchvision
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected PyTorch CUDA 13.0, found {torch.version.cuda!r}.")

    cudnn_version = torch.backends.cudnn.version()
    if not isinstance(cudnn_version, int) or cudnn_version < 90_000:
        raise RuntimeError(f"Expected loadable cuDNN 9 or newer, found {cudnn_version!r}.")

    cudnn_distribution = metadata.distribution("nvidia-cudnn-cu13")
    cudnn_library = Path(
        cudnn_distribution.locate_file("nvidia/cudnn/lib/libcudnn.so.9")
    )
    if not cudnn_library.is_file():
        raise RuntimeError(f"Expected packaged cuDNN library at {cudnn_library}.")
    ctypes.CDLL(str(cudnn_library))

    has_nvidia_devices = Path("/dev/nvidiactl").exists() and any(
        Path("/dev").glob("nvidia[0-9]*")
    )
    if not has_nvidia_devices and torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() must be false when no NVIDIA devices exist.")

    print(
        f"Verified torch CUDA {torch.version.cuda}, cuDNN {cudnn_version}, "
        f"GPU available={torch.cuda.is_available()}."
    )


def main() -> None:
    assert_exact_environment()
    assert_distribution_metadata()
    assert_install_report()
    assert_xformers_install_report()
    assert_torch_unchanged_by_xformers()
    assert_xformers_import_and_info()
    assert_gpu_verifier()
    assert_cuda_and_cudnn()
    print("PyTorch and xformers image verification passed.")


if __name__ == "__main__":
    main()
