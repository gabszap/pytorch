ARG IMAGE_BASE="ghcr.io/gabszap/python:3.12-v3-cuda-13.0.3-cudnn-runtime-24.04"
FROM ${IMAGE_BASE}

LABEL org.opencontainers.image.source="https://github.com/gabszap/pytorch" \
      org.opencontainers.image.url="https://github.com/gabszap/pytorch" \
      org.opencontainers.image.title="gabszap/pytorch" \
      org.opencontainers.image.version="2.13.0-py3.12-v3-cuda-13.0.3-cudnn-runtime-24.04" \
      org.opencontainers.image.description="Unofficial NVIDIA CUDA PyTorch image with xformers 0.0.35, derived from AI-Dock PyTorch" \
      org.opencontainers.image.base.name="ghcr.io/gabszap/python:3.12-v3-cuda-13.0.3-cudnn-runtime-24.04" \
      org.opencontainers.image.vendor="gabszap; derived from AI-Dock by Rob Ballantyne" \
      org.opencontainers.image.licenses="LicenseRef-AI-Dock-Custom" \
      maintainer="gabszap"

ENV IMAGE_SLUG="pytorch"

# Copy early so we can use scripts in the build - Changes to these files will invalidate the cache and cause a rebuild.
COPY --chown=0:1111 ./COPY_ROOT_0/ /

ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}
ARG PYTHON_VENV_NAME=python_312
ENV PYTHON_VENV_NAME=${PYTHON_VENV_NAME}
ENV PYTHON_DEFAULT_VENV=${PYTHON_VENV_NAME}
ARG PYTORCH_VERSION=2.13.0+cu130
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ARG TORCHVISION_VERSION=0.28.0+cu130
ENV TORCHVISION_VERSION=${TORCHVISION_VERSION}
ARG TORCHAUDIO_VERSION=2.11.0+cu130
ENV TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ENV PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}
ARG XFORMERS_VERSION=0.0.35
ENV XFORMERS_VERSION=${XFORMERS_VERSION}
ARG XFORMERS_INDEX_URL=https://download.pytorch.org/whl/cu130
ENV XFORMERS_INDEX_URL=${XFORMERS_INDEX_URL}
ARG XFORMERS_WHEEL=xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl
ENV XFORMERS_WHEEL=${XFORMERS_WHEEL}
ARG XFORMERS_WHEEL_SHA256=962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d
ENV XFORMERS_WHEEL_SHA256=${XFORMERS_WHEEL_SHA256}

RUN set -eo pipefail && /opt/ai-dock/bin/build/layer0/init.sh | tee /var/log/build.log

# Keep init.sh as-is and place additional logic in /opt/ai-dock/bin/preflight.sh
CMD ["init.sh"]
