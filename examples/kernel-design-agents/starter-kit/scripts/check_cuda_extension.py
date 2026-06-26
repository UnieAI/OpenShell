"""
Compile-smoke helper for CUDA + Torch binding submissions.

This is a fast intermediate validation stage before the full FlashInfer local
benchmark sweep. It verifies that the active CUDA source can be built through
torch.utils.cpp_extension.load().
"""

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import torch
from torch.utils.cpp_extension import load


PROJECT_ROOT = Path(__file__).parent.parent


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(log_path):
    if not log_path:
        yield
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, handle)
    sys.stderr = TeeStream(original_stderr, handle)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        handle.close()


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def parse_flags(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(" ") if item]


def resolve_cuda_source(build_config: dict) -> Path:
    source_dir_name = build_config.get("source_dir") or "cuda"
    entry_point = build_config["entry_point"]
    source_file = entry_point.split("::", 1)[0] if ".cu::" in entry_point else "kernel.cu"
    return PROJECT_ROOT / "solution" / source_dir_name / source_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile-smoke the active CUDA torch extension")
    parser.add_argument("--name", default="kda_cuda_smoke")
    parser.add_argument("--log-file", default=os.environ.get("KDA_CUDA_EXTENSION_LOG_PATH", ""))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--extra-cflags", default=os.environ.get("KDA_CUDA_EXTENSION_EXTRA_CFLAGS", "-O0"))
    parser.add_argument("--extra-cuda-cflags", default=os.environ.get("KDA_CUDA_EXTENSION_EXTRA_CUDA_CFLAGS", "-O0"))
    return parser


def main():
    args = build_parser().parse_args()
    with tee_output(args.log_file or None):
        config = load_config()
        build_config = config["build"]

        language = build_config.get("language")
        binding = build_config.get("binding", "tvm-ffi")
        if language != "cuda":
            print(f"Skipping compile smoke: language={language!r} is not CUDA.")
            return
        if binding != "torch":
            print(f"Skipping compile smoke: binding={binding!r} is not torch.")
            return

        source_path = resolve_cuda_source(build_config)
        if not source_path.exists():
            raise FileNotFoundError(f"CUDA source file not found: {source_path}")

        print(f"Compiling CUDA source: {source_path.relative_to(PROJECT_ROOT)}")
        print(f"Extension name: {args.name}")
        print(f"extra_cflags={args.extra_cflags!r}")
        print(f"extra_cuda_cflags={args.extra_cuda_cflags!r}")

        module = load(
            name=args.name,
            sources=[str(source_path)],
            verbose=args.verbose,
            extra_cflags=parse_flags(args.extra_cflags),
            extra_cuda_cflags=parse_flags(args.extra_cuda_cflags),
        )
        print(f"Extension compiled successfully: {module.__name__}")


if __name__ == "__main__":
    main()
