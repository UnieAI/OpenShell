import importlib
import os
import shutil
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class CapabilityStatus:
    name: str
    available: bool
    detail: str


class RuntimeCapabilities:
    def __init__(self, statuses: Iterable[CapabilityStatus]):
        self._statuses = {status.name: status for status in statuses}

    def get(self, name: str) -> CapabilityStatus:
        return self._statuses[name]

    def require(self, *names: str) -> None:
        missing = [self._statuses[name] for name in names if not self._statuses[name].available]
        if not missing:
            return

        reasons = "; ".join(f"{status.name}: {status.detail}" for status in missing)
        raise RuntimeError(f"Runtime capability check failed: {reasons}")

    def summary(self) -> str:
        parts = []
        for name in sorted(self._statuses):
            status = self._statuses[name]
            state = "ok" if status.available else "missing"
            parts.append(f"{name}={state} ({status.detail})")
        return ", ".join(parts)


def _probe_import(module_name: str) -> CapabilityStatus:
    try:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", "<built-in>")
        return CapabilityStatus(module_name, True, module_file)
    except Exception as exc:
        return CapabilityStatus(module_name, False, str(exc))


def _probe_command(command_name: str, explicit_path: Optional[str] = None) -> CapabilityStatus:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    resolved = shutil.which(command_name)
    if resolved:
        candidates.append(resolved)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return CapabilityStatus(command_name, True, candidate)
    return CapabilityStatus(command_name, False, "not found in PATH")


def detect_runtime_capabilities(settings=None) -> RuntimeCapabilities:
    codex_explicit = None
    ncu_explicit = None
    if settings and getattr(settings, "codex_command", None):
        codex_explicit = settings.codex_command[0]
    if settings and getattr(settings, "ncu_command", None):
        ncu_explicit = settings.ncu_command[0]

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc_explicit = os.path.join(cuda_home, "bin", "nvcc")

    statuses = [
        _probe_import("torch"),
        _probe_import("tvm_ffi"),
        _probe_command("nvcc", nvcc_explicit),
        _probe_command("ncu", ncu_explicit),
        _probe_command("codex", codex_explicit),
    ]
    return RuntimeCapabilities(statuses)


def require_compile_capabilities(capabilities: RuntimeCapabilities) -> None:
    capabilities.require("torch", "tvm_ffi", "nvcc")


def require_verify_capabilities(capabilities: RuntimeCapabilities) -> None:
    capabilities.require("torch", "tvm_ffi")


def require_profile_capabilities(capabilities: RuntimeCapabilities) -> None:
    capabilities.require("torch", "tvm_ffi", "nvcc", "ncu")


def require_optimize_capabilities(capabilities: RuntimeCapabilities) -> None:
    capabilities.require("torch", "tvm_ffi", "nvcc", "ncu", "codex")
