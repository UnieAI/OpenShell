import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml


def resolve_user_path(path: Union[str, os.PathLike], base: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_optional_path(raw_value, default: Path, base: Path) -> Path:
    if raw_value in (None, ""):
        return default
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".autoagent-write-test"
        with probe.open("w", encoding="utf-8") as handle:
            handle.write("")
        probe.unlink()
        return True
    except OSError:
        return False


def _select_writable_directory(*candidates: Path) -> Path:
    for candidate in candidates:
        if _is_writable_directory(candidate):
            return candidate
    raise PermissionError(
        f"No writable state directory found among: {', '.join(str(path) for path in candidates)}"
    )


def _ensure_writable_path(candidate: Path, fallback: Path) -> Path:
    parent = candidate if candidate.suffix == "" else candidate.parent
    if _is_writable_directory(parent):
        return candidate

    fallback_parent = fallback if fallback.suffix == "" else fallback.parent
    if _is_writable_directory(fallback_parent):
        return fallback

    raise PermissionError(
        f"Neither '{candidate}' nor fallback '{fallback}' is writable"
    )


def _normalize_command(raw_value) -> tuple[str, ...]:
    if raw_value is None:
        return ("codex",)
    if isinstance(raw_value, str):
        return tuple(shlex.split(raw_value)) or ("codex",)
    if isinstance(raw_value, list):
        return tuple(str(part) for part in raw_value)
    raise ValueError(f"Unsupported command format: {raw_value!r}")


def _normalize_ncu_sets(raw_value) -> tuple[str, ...]:
    if raw_value is None:
        return ("full",)
    if isinstance(raw_value, str):
        return (raw_value,)
    if isinstance(raw_value, list):
        return tuple(str(item) for item in raw_value)
    raise ValueError(f"Unsupported NCU set format: {raw_value!r}")


@dataclass(frozen=True)
class RuntimeSettings:
    workspace: Path
    state_dir: Path
    runtime_config_path: Path
    template_path: Path
    output_cu: Path
    output_so: Path
    active_kernel_yaml: Path
    report_dir: Path
    kernels_dir: Path
    output_dir: Path
    llm_backend: str
    profile_executor: str
    python_command: tuple[str, ...]
    codex_command: tuple[str, ...]
    ncu_command: tuple[str, ...]
    ncu_sets: tuple[str, ...]

    def load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def dump_yaml(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)


def load_runtime_settings(
    runtime_config_path: Optional[str] = None,
    workspace: Optional[str] = None,
    state_dir: Optional[str] = None,
) -> RuntimeSettings:
    default_workspace = Path(
        workspace or os.environ.get("AUTOAGENT_WORKSPACE") or Path.cwd()
    ).resolve()

    runtime_config = resolve_user_path(
        runtime_config_path or os.environ.get("AUTOAGENT_RUNTIME_CONFIG") or "config.yaml",
        default_workspace,
    )
    config = _read_yaml(runtime_config)

    resolved_workspace = default_workspace
    requested_state_dir = Path(
        state_dir
        or os.environ.get("AUTOAGENT_STATE_DIR")
        or config.get("STATE_DIR")
        or resolved_workspace
    )
    if not requested_state_dir.is_absolute():
        requested_state_dir = resolved_workspace / requested_state_dir
    resolved_state_dir = _select_writable_directory(
        requested_state_dir.resolve(),
        Path("/sandbox/autoagent"),
        Path("/tmp/autoagent"),
    )

    template_path = _resolve_optional_path(
        config.get("TEMPLATE_PATH"),
        resolved_workspace / "kernel_template.cu.j2",
        resolved_workspace,
    )
    output_cu = _resolve_optional_path(
        config.get("OUTPUT_CU"),
        resolved_state_dir / "kernel.cu",
        resolved_state_dir,
    )
    output_cu = _ensure_writable_path(output_cu, resolved_state_dir / "kernel.cu")
    output_so = _resolve_optional_path(
        config.get("OUTPUT_SO"),
        resolved_state_dir / "kernel.so",
        resolved_state_dir,
    )
    output_so = _ensure_writable_path(output_so, resolved_state_dir / "kernel.so")
    active_kernel_yaml = _resolve_optional_path(
        config.get("ACTIVE_KERNEL_YAML"),
        resolved_state_dir / "active_kernel.yaml",
        resolved_state_dir,
    )
    active_kernel_yaml = _ensure_writable_path(
        active_kernel_yaml,
        resolved_state_dir / "active_kernel.yaml",
    )
    report_dir = _resolve_optional_path(
        config.get("REPORT_DIR"),
        resolved_state_dir / "ncu_reports",
        resolved_state_dir,
    )
    report_dir = _ensure_writable_path(report_dir, resolved_state_dir / "ncu_reports")
    kernels_dir = _resolve_optional_path(
        config.get("KERNELS_DIR"),
        resolved_state_dir / "kernels",
        resolved_state_dir,
    )
    kernels_dir = _ensure_writable_path(kernels_dir, resolved_state_dir / "kernels")
    output_dir = _resolve_optional_path(
        config.get("OUTPUT_DIR"),
        resolved_state_dir / "output",
        resolved_state_dir,
    )
    output_dir = _ensure_writable_path(output_dir, resolved_state_dir / "output")

    report_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    return RuntimeSettings(
        workspace=resolved_workspace,
        state_dir=resolved_state_dir,
        runtime_config_path=runtime_config,
        template_path=template_path,
        output_cu=output_cu,
        output_so=output_so,
        active_kernel_yaml=active_kernel_yaml,
        report_dir=report_dir,
        kernels_dir=kernels_dir,
        output_dir=output_dir,
        llm_backend=os.environ.get("AUTOAGENT_BACKEND")
        or config.get("LLM_BACKEND", "codex-exec"),
        profile_executor=os.environ.get("AUTOAGENT_PROFILE_EXECUTOR")
        or config.get("PROFILE_EXECUTOR", "disabled"),
        python_command=_normalize_command(
            os.environ.get("AUTOAGENT_PYTHON_COMMAND")
            or config.get("PYTHON_COMMAND")
            or sys.executable
        ),
        codex_command=_normalize_command(
            os.environ.get("AUTOAGENT_CODEX_COMMAND")
            or config.get("CODEX_COMMAND")
        ),
        ncu_command=_normalize_command(
            os.environ.get("AUTOAGENT_NCU_COMMAND")
            or config.get("NCU_COMMAND")
            or "ncu"
        ),
        ncu_sets=_normalize_ncu_sets(config.get("NCU_SET")),
    )
