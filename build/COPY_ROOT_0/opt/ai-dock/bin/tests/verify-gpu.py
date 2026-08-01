#!/opt/environments/python/python_312/bin/python

import argparse
import subprocess
import sys
import traceback
from typing import Any, NamedTuple


class VerificationError(RuntimeError):
    pass


class TestSizes(NamedTuple):
    matmul: int
    batch: int
    sequence: int
    heads: int
    head_dim: int


def bounded_integer(name: str, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed_value = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from error

        if not 1 <= parsed_value <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between 1 and {maximum}, got {parsed_value}"
            )
        return parsed_value

    return parse


def parse_capability(value: str) -> tuple[int, int]:
    capability_parts = value.split(".")
    if len(capability_parts) != 2 or not all(part.isdigit() for part in capability_parts):
        raise argparse.ArgumentTypeError(
            f"expected capability must use MAJOR.MINOR, got {value!r}"
        )
    return int(capability_parts[0]), int(capability_parts[1])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded PyTorch and xformers CUDA checks inside the current container. "
            "This verifier fails when no CUDA GPU is available."
        )
    )
    parser.add_argument(
        "--expected-capability",
        type=parse_capability,
        metavar="MAJOR.MINOR",
        help="require an exact CUDA compute capability, for example 12.0",
    )
    parser.add_argument(
        "--matmul-size",
        type=bounded_integer("matmul size", 1024),
        default=256,
        metavar="N",
        help="side length for FP16/BF16 square matmuls (default: 256; maximum: 1024)",
    )
    parser.add_argument(
        "--batch-size",
        type=bounded_integer("batch size", 4),
        default=1,
        metavar="N",
        help="attention batch size (default: 1; maximum: 4)",
    )
    parser.add_argument(
        "--sequence-length",
        type=bounded_integer("sequence length", 512),
        default=128,
        metavar="N",
        help="attention sequence length (default: 128; maximum: 512)",
    )
    parser.add_argument(
        "--heads",
        type=bounded_integer("head count", 16),
        default=4,
        metavar="N",
        help="attention head count (default: 4; maximum: 16)",
    )
    parser.add_argument(
        "--head-dim",
        type=bounded_integer("head dimension", 256),
        default=64,
        metavar="N",
        help="attention head dimension (default: 64; maximum: 256)",
    )
    return parser.parse_args()


def print_environment(torch: Any, xformers: Any) -> None:
    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"xformers={xformers.__version__}", flush=True)
    print(f"torch_cuda={torch.version.cuda}", flush=True)
    print(f"cudnn={torch.backends.cudnn.version()}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)


def require_cuda_device(
    torch: Any, expected_capability: tuple[int, int] | None
) -> tuple[int, tuple[int, int], str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise VerificationError(
            "No CUDA GPU is available. Run this explicit GPU verifier in a container "
            "with an NVIDIA GPU exposed."
        )

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    architecture = f"sm_{capability[0]}{capability[1]}"
    print(f"cuda_device={device}", flush=True)
    print(f"cuda_device_name={torch.cuda.get_device_name(device)}", flush=True)
    print(f"cuda_capability={capability[0]}.{capability[1]}", flush=True)
    print(f"torch_cuda_arch_list={torch.cuda.get_arch_list()}", flush=True)

    if expected_capability is not None:
        expected_text = f"{expected_capability[0]}.{expected_capability[1]}"
        print(f"expected_capability={expected_text}", flush=True)
        if capability != expected_capability:
            actual_text = f"{capability[0]}.{capability[1]}"
            raise VerificationError(
                f"Expected CUDA capability {expected_text}, found {actual_text}."
            )

    if architecture not in torch.cuda.get_arch_list():
        raise VerificationError(
            f"Device architecture {architecture} is absent from the PyTorch wheel "
            f"architecture list {torch.cuda.get_arch_list()}."
        )

    return device, capability, architecture


def assert_finite_tensor(torch: Any, tensor: Any, test_name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise VerificationError(f"{test_name} returned non-finite values.")


def run_matmul_checks(torch: Any, size: int) -> None:
    for dtype in (torch.float16, torch.bfloat16):
        left = torch.randn((size, size), device="cuda", dtype=dtype)
        right = torch.randn((size, size), device="cuda", dtype=dtype)
        result = left @ right
        torch.cuda.synchronize()
        assert_finite_tensor(torch, result, f"{dtype} matmul")
        print(
            f"matmul_pass dtype={dtype} shape={tuple(result.shape)}",
            flush=True,
        )


def run_sdpa_check(torch: Any, sizes: TestSizes) -> None:
    import torch.nn.functional as torch_functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    shape = (sizes.batch, sizes.heads, sizes.sequence, sizes.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with sdpa_kernel(SDPBackend.MATH):
        result = torch_functional.scaled_dot_product_attention(query, key, value)
    torch.cuda.synchronize()
    if result.shape != query.shape:
        raise VerificationError(
            f"PyTorch SDPA returned shape {tuple(result.shape)}, expected {tuple(query.shape)}."
        )
    assert_finite_tensor(torch, result, "PyTorch SDPA")
    print(
        f"sdpa_pass backend=MATH(forced) dtype={result.dtype} shape={tuple(result.shape)}",
        flush=True,
    )


def print_xformers_info() -> None:
    print("xformers_info_begin", file=sys.stderr, flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "xformers.info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as error:
        print(
            f"Unable to run {sys.executable} -m xformers.info: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
    else:
        if result.stdout:
            print(result.stdout.rstrip(), file=sys.stderr, flush=True)
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        print(f"xformers_info_exit_code={result.returncode}", file=sys.stderr, flush=True)
    print("xformers_info_end", file=sys.stderr, flush=True)


def run_xformers_check(torch: Any, xformers_ops: Any, sizes: TestSizes) -> str:
    from xformers.ops.fmha.common import Inputs
    from xformers.ops.fmha.dispatch import _dispatch_fw

    shape = (sizes.batch, sizes.sequence, sizes.heads, sizes.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    inputs = Inputs(query=query, key=key, value=value, p=0.0)

    try:
        for operator in xformers_ops.fmha.ALL_FW_OPS:
            try:
                unsupported_reasons = operator.not_supported_reasons(inputs)
                status = "supported" if not unsupported_reasons else "; ".join(unsupported_reasons)
            except Exception as error:
                status = f"diagnostic failed: {type(error).__name__}: {error}"
            print(f"xformers_backend {operator.NAME}: {status}", flush=True)

        selected_operator = _dispatch_fw(inputs, needs_gradient=False)
        print(f"xformers_selected_backend={selected_operator.NAME}", flush=True)
        result = xformers_ops.memory_efficient_attention(
            query,
            key,
            value,
            op=(selected_operator, None),
        )
        torch.cuda.synchronize()
        if result.shape != query.shape:
            raise VerificationError(
                f"xformers attention returned shape {tuple(result.shape)}, "
                f"expected {tuple(query.shape)}."
            )
        assert_finite_tensor(torch, result, "xformers memory_efficient_attention")
    except Exception as error:
        print(
            f"xformers_exception={type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        print_xformers_info()
        raise VerificationError("xformers memory_efficient_attention failed.") from error

    print(
        f"xformers_pass dtype={result.dtype} layout=BMHK shape={tuple(result.shape)} "
        f"backend={selected_operator.NAME}",
        flush=True,
    )
    return selected_operator.NAME


def run_verification(arguments: argparse.Namespace) -> None:
    try:
        import torch
        import xformers
        import xformers.ops as xformers_ops
    except Exception as error:
        raise VerificationError(
            f"Unable to import the GPU test dependencies: {type(error).__name__}: {error}"
        ) from error

    print_environment(torch, xformers)
    _, capability, architecture = require_cuda_device(torch, arguments.expected_capability)
    sizes = TestSizes(
        matmul=arguments.matmul_size,
        batch=arguments.batch_size,
        sequence=arguments.sequence_length,
        heads=arguments.heads,
        head_dim=arguments.head_dim,
    )

    torch.manual_seed(0)
    with torch.no_grad():
        run_matmul_checks(torch, sizes.matmul)
        run_sdpa_check(torch, sizes)
        selected_backend = run_xformers_check(torch, xformers_ops, sizes)

    print(
        f"PASS: GPU validation succeeded; capability={capability[0]}.{capability[1]}, "
        f"architecture={architecture}, xformers_backend={selected_backend}.",
        flush=True,
    )


def main() -> int:
    arguments = parse_arguments()
    try:
        run_verification(arguments)
    except VerificationError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
