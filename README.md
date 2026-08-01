# gabszap/pytorch

> [!IMPORTANT]
> This repository is a personally maintained, modified fork of
> [ai-dock/pytorch](https://github.com/ai-dock/pytorch). It is not an official
> AI-Dock project and is not endorsed by AI-Dock or Rob Ballantyne.

This image layers the official PyTorch CUDA 13.0 wheels onto
[`gabszap/python`](https://github.com/gabszap/python) while retaining the
AI-Dock runtime, Jupyter integration, ports, and storage layout. It targets
NVIDIA GPUs on Linux amd64, including RTX 50-series/Blackwell GPUs with
compute capability `sm_120`.

## Exact stack

| Component | Version/source |
| --- | --- |
| Base image | `ghcr.io/gabszap/python:3.12-v3-cuda-13.0.3-cudnn-runtime-24.04` |
| Ubuntu | 24.04 |
| Python/default venv | 3.12 / `/opt/environments/python/python_312` |
| PyTorch | `torch==2.13.0+cu130` |
| TorchVision | `torchvision==0.28.0+cu130` |
| TorchAudio | `torchaudio==2.11.0+cu130` |
| xformers | `xformers==0.0.35` (`xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl`) |
| Wheel index | `https://download.pytorch.org/whl/cu130` |

All four components are exact pins from the official PyTorch CUDA 13.0 index.
xformers is installed after the complete torch stack with `--no-deps`, and its
official wheel SHA-256 is
`962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d`.
The official stable-ABI wheel metadata declares Python `>=3.9`, includes
Python 3.12 in its classifiers, and declares `torch>=2.10`; the image retains
the exact `torch==2.13.0+cu130` stack and existing NumPy version rather than
allowing dependency changes.

The image intentionally does **not** include the separate flash-attn,
flash-attn-3, flash-attn-4, or SageAttention distributions. Compatible
official CUDA 13/sm120 wheels are not part of this pinned stack, and installing
them could replace torch or introduce unvalidated compiled kernels. xformers
contains its own registered attention backends; that does not mean those
separate extension packages are installed.

The supported image platform is NVIDIA-only Linux amd64. The official cu130
index also supplies aarch64 wheels, but this repository does not build or test
an aarch64 image and Compose explicitly pins `linux/amd64`. CPU and AMD/ROCm
variants are not provided.

## Image tags

The versioned public GHCR image is:

```text
ghcr.io/gabszap/pytorch:2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04
```

The manual build workflow publishes that immutable version convention plus
`latest-cuda` and `latest` to `ghcr.io/gabszap/pytorch`.

> [!NOTE]
> The base Python image must already be published and publicly pullable at
> `ghcr.io/gabszap/python:3.12-v3-cuda-13.0.3-cudnn-runtime-24.04` before the
> GitHub Actions build can run. A local copy is sufficient only for local
> Docker builds, not for GitHub-hosted runners.

## Host requirements

- Linux amd64 and an NVIDIA GPU for CUDA workloads.
- NVIDIA Container Toolkit configured for Docker.
- A CUDA 13-compatible host driver. NVIDIA R580 is the minimum driver family;
  use a current R580-series or newer compatible production driver. CUDA 13.0
  GA lists Linux driver `580.65.06` as its minimum patch level, and R580 or
  newer is the recommended baseline.
- Enough disk space for the CUDA runtime, official wheels, and their declared
  NVIDIA dependencies.

RTX 50-series devices report compute capability `(12, 0)`. The cu130 PyTorch
build includes `sm_120` support; actual GPU execution still must be validated
on the deployment host. xformers 0.0.35 imports and reports its registered
backends in a no-GPU build, but RTX 50/sm120 kernel execution remains pending
real GPU validation. Individual xformers backends may reject sm120, so the
in-image verifier reports support reasons and the backend selected for its
standard BMHK tensors. Native PyTorch scaled dot product attention (SDPA)
remains the recommended baseline and fallback.

### Verified hardware results

Tested on NVIDIA GeForce RTX 5060 Ti 16 GB with driver 595.84 and host maximum
CUDA 13.2. Device reported compute capability `(12,0)` (sm_120).
Using `torch==2.13.0+cu130` and `xformers==0.0.35`:
- FP16/BF16 matmul passed.
- Native SDPA baseline passed.
- `xformers.ops.memory_efficient_attention` passed, with `fa2F@2.5.7-pt`
  selected as the backend; CUTLASS/FA3 were unavailable for sm_120.

This result applies only to this tested hardware/driver combination. This image
does not include a separate Flash-Attention distribution. Native PyTorch
scaled dot product attention (SDPA) remains the recommended baseline and
fallback.

## Build and run

Build locally from the published base:

```bash
docker compose build
```

Or build directly with every exact boundary shown:

```bash
docker build \
  --build-arg IMAGE_BASE=ghcr.io/gabszap/python:3.12-v3-cuda-13.0.3-cudnn-runtime-24.04 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg PYTHON_VENV_NAME=python_312 \
  --build-arg PYTORCH_VERSION=2.13.0+cu130 \
  --build-arg TORCHVISION_VERSION=0.28.0+cu130 \
  --build-arg TORCHAUDIO_VERSION=2.11.0+cu130 \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  --build-arg XFORMERS_VERSION=0.0.35 \
  --build-arg XFORMERS_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  --build-arg XFORMERS_WHEEL=xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl \
  --build-arg XFORMERS_WHEEL_SHA256=962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d \
  --tag ghcr.io/gabszap/pytorch:2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04 \
  build
```

Uncomment the NVIDIA device reservation in `docker-compose.yaml`, then run:

```bash
docker compose up
```

The inherited SSH, service portal, Jupyter, Syncthing ports, workspace
volume, default `python_312` kernel, and supervisor behavior remain available.
JupyterLab is the default UI.

## Validation

The Docker build runs package imports, exact local-version checks, `pip
check`, CUDA/cuDNN metadata and library-load checks, and source-report checks.
It verifies the dedicated xformers pip report, exact URL and hash, Python/torch
metadata, and unchanged torch versions before and after the `--no-deps`
install. It also records `python -m xformers.info` output at
`/opt/ai-dock/share/xformers-info.txt` and asserts that FlashAttention and
SageAttention distributions remain absent. Building does not require a GPU.
On a builder without NVIDIA devices, `torch.cuda.is_available()` is expected
to be false. xformers import/info success in that context validates packaging
and backend registration only; it does not claim that a CUDA kernel executed.

Run the repository's isolated runtime check against a local image:

```bash
build/tests/runtime-image.sh ghcr.io/gabszap/pytorch:2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04
```

Run GPU validation when a GPU is available. The optional second argument
enforces an exact capability, for example RTX 50-series `(12,0)`:

```bash
build/tests/gpu-validation.sh ghcr.io/gabszap/pytorch:2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04 12.0
```

The expected capability can instead be supplied through
`EXPECTED_CAPABILITY=12.0`; the host wrapper maps it to the verifier's
`--expected-capability` option.

### Vast.ai direct-container validation

The verifier is installed in the image, so a Vast.ai user can run it directly
inside the rented container without a Docker daemon or repository checkout:

```bash
/opt/environments/python/python_312/bin/python \
  /opt/ai-dock/bin/tests/verify-gpu.py \
  --expected-capability 12.0
```

An RTX 5060 Ti is sufficient. The bounded defaults use small FP16/BF16
matrices and attention tensors and fit comfortably within its 8 GB VRAM. The
Vast host driver must support a CUDA 13 container (reported maximum CUDA 13.x)
and use the R580 driver family or newer.

When a repository checkout and Docker daemon are available on the host, the
equivalent wrapper command is:

```bash
EXPECTED_CAPABILITY=12.0 build/tests/gpu-validation.sh \
  ghcr.io/gabszap/pytorch:2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04
```

The in-image script runs FP16 and BF16 CUDA matrix multiplication, forced
native PyTorch SDPA, and
`xformers.ops.memory_efficient_attention`; it prints the selected xformers
backend and support diagnostics and fails when no backend executes. It reports
an error and exits nonzero when no GPU is exposed. Only the host wrapper reports
`SKIP` without claiming success when no host GPU is available. Equivalent basic
direct checks are:

```bash
docker run --rm --gpus all ghcr.io/gabszap/pytorch:2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04 \
  /opt/environments/python/python_312/bin/python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())'

docker run --rm --gpus all ghcr.io/gabszap/pytorch:2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04 \
  /opt/environments/python/python_312/bin/python -c \
  'import torch; a=torch.randn(256,256,device="cuda"); b=a@a; torch.cuda.synchronize(); print(b.shape, b.device)'
```

For an RTX 50-series host, the capability result is expected to be `(12, 0)`.
These commands are instructions, not a claim that GPU validation was run for
any published image.

## Upstream, license, and attribution

This fork retains the AI-Dock filesystem layout, runtime integration,
branding, notices, funding links, and attribution. Review
[LICENSE.md](LICENSE.md), [NOTICE.md](NOTICE.md), and
[FUNDING.yml](.github/FUNDING.yml) before use or redistribution. Distribution
of this derivative is undertaken with the explicit permission required by the
custom AI-Dock license. The complete modified source is maintained publicly
in this repository.

AI-Dock base behavior is documented in the
[base-image wiki](https://github.com/ai-dock/base-image/wiki). The upstream
project describes cloud use with [Vast.ai](https://link.ai-dock.org/vast.ai)
and [Runpod.io](https://link.ai-dock.org/runpod.io). These original links are
retained for attribution and historical reference; they do not describe this
fork's image:

- [Upstream PyTorch package](https://github.com/ai-dock/pytorch/pkgs/container/pytorch)
- [Upstream Vast.ai PyTorch template](https://link.ai-dock.org/template-vast-pytorch)
- [Upstream Vast.ai alternate-accelerator template](https://link.ai-dock.org/template-vast-pytorch-rocm)
- [Upstream Runpod.io PyTorch template](https://link.ai-dock.org/template-runpod-pytorch)
- Upstream workflow: [![Upstream Docker Build](https://github.com/ai-dock/pytorch/actions/workflows/docker-build.yml/badge.svg)](https://github.com/ai-dock/pytorch/actions/workflows/docker-build.yml)

The original author,
[@robballantyne](https://github.com/robballantyne), may be compensated if you
sign up to services linked in the upstream materials. Testing GPU image
variants is costly and time-consuming; upstream
[AI-Dock sponsorships](https://github.com/sponsors/ai-dock) help support that
work.
