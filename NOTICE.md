## Notice:

This repository is a personally maintained, modified derivative of
[AI-Dock PyTorch](https://github.com/ai-dock/pytorch), originally authored by
Robert Ballantyne. It is not an official AI-Dock project. Public distribution
of this derivative is undertaken with the explicit permission required by the
custom license. The original notice follows unchanged.

The derivative image bundles `xformers==0.0.35` from the official PyTorch
CUDA 13.0 wheel index. Its exact Linux amd64 wheel is
`xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl` with SHA-256
`962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d`.
The official stable-ABI wheel metadata declares Python `>=3.9` and
`torch>=2.10`; this image pins and preserves `torch==2.13.0+cu130`. RTX
50/sm120 xformers kernel execution is pending real GPU validation and is
expected to use the Triton fallback where
CUTLASS/FA2/FA3 backends do not support sm120. Native PyTorch SDPA remains the
recommended fallback. Separate FlashAttention and SageAttention extensions
are intentionally excluded because compatible official CUDA 13/sm120 wheels
are not included in this exact stack and could alter its pinned dependencies.

I have chosen to apply a custom license to this software for the following reasons:

- **Uniqueness of Containers:** Common open-source licenses may not adequately address the nuances of software distributed within containers. My custom license ensures clarity regarding the separation of my code from bundled software, thereby respecting the rights of other authors.

- **Preservation of Source Code Integrity:** I am committed to maintaining the integrity of the source code while adhering to the spirit of open-source software. My custom license helps ensure transparency and accountability in my development practices.

- **Funding and Control of Distribution:** Some of the funding for this project comes from maintaining control of distribution. This funding model wouldn't be possible without limiting distribution in certain ways, ultimately supporting the project's mission.

- **Empowering Access:** Supported by controlled distribution, the mission of this project is to empower users with access to valuable tools and resources in the cloud, enabling them to utilize software that may otherwise require hardware resources beyond their reach.

I welcome sponsorship from commercial entities utilizing this software, although it is not mandatory. Your support helps sustain the ongoing development and improvement of this project.

You can sponsor this project at https://github.com/sponsors/ai-dock.

Your understanding and support are greatly appreciated.
