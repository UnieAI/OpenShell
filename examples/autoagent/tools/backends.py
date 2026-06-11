import re
import subprocess
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

import yaml
from tools.capabilities import detect_runtime_capabilities


@dataclass(frozen=True)
class GenerationRequest:
    kernel_code: str
    optimization_text: str
    meta: dict
    version_label: str
    output_meta_path: Path
    kernel_config_path: Path
    error_context: Optional[str] = None
    strategy_hint: Optional[str] = None


class KernelGenerationBackend:
    name = "unknown"

    def describe_unavailability(self) -> Optional[str]:
        return None

    def generate(self, request: GenerationRequest) -> bool:
        raise NotImplementedError


class CodexExecBackend(KernelGenerationBackend):
    name = "codex-exec"

    def __init__(self, settings):
        self.settings = settings
        self.scratch_dir = settings.state_dir / "tmp"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def describe_unavailability(self) -> Optional[str]:
        status = detect_runtime_capabilities(self.settings).get("codex")
        if not status.available:
            return f"the 'codex' runtime capability is unavailable: {status.detail}"
        return None

    def _resolve_runtime_root(self) -> Path:
        override = os.environ.get("AUTOAGENT_CODEX_RUNTIME_ROOT")
        if override:
            return Path(override).resolve()
        return self.settings.state_dir / "tmp"

    def _build_codex_env(self) -> dict[str, str]:
        runtime_root = self._resolve_runtime_root()
        home_dir = runtime_root / "home"
        tmp_dir = runtime_root / "tmp"
        xdg_cache_dir = runtime_root / "cache"
        xdg_config_dir = runtime_root / "config"
        xdg_data_dir = runtime_root / "data"
        xdg_state_dir = runtime_root / "state"
        codex_home_dir = runtime_root / "codex"
        codex_sqlite_dir = codex_home_dir / "sqlite"
        codex_logs_dir = codex_home_dir / "logs"

        for path in (
            runtime_root,
            home_dir,
            tmp_dir,
            xdg_cache_dir,
            xdg_config_dir,
            xdg_data_dir,
            xdg_state_dir,
            codex_home_dir,
            codex_sqlite_dir,
            codex_logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home_dir),
                "TMPDIR": str(tmp_dir),
                "TMP": str(tmp_dir),
                "TEMP": str(tmp_dir),
                "XDG_CACHE_HOME": str(xdg_cache_dir),
                "XDG_CONFIG_HOME": str(xdg_config_dir),
                "XDG_DATA_HOME": str(xdg_data_dir),
                "XDG_STATE_HOME": str(xdg_state_dir),
                "CODEX_HOME": str(codex_home_dir),
                "CODEX_SQLITE_HOME": str(codex_sqlite_dir),
            }
        )
        return env

    def generate(self, request: GenerationRequest) -> bool:
        reason = self.describe_unavailability()
        if reason:
            raise RuntimeError(reason)

        with request.kernel_config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        kernel_name = config.get("kernel_name", "kernel")
        description = config.get("description", "")
        parameters = config.get("parameters", [])

        sig_args = []
        param_yaml_lines = []
        for parameter in parameters:
            parameter_type = parameter["type"]
            parameter_name = parameter["name"]
            if parameter_type == "tensor":
                sig_args.append(f"float* {parameter_name}")
                param_yaml_lines.append(f"    - {{name: {parameter_name}, type: tensor}}")
            else:
                sig_args.append(f"int {parameter_name}")
                param_yaml_lines.append(f"    - {{name: {parameter_name}, type: int}}")

        signature = f"__global__ void {kernel_name}(" + ", ".join(sig_args) + ")"
        param_yaml_str = "\n".join(param_yaml_lines)

        error_section = ""
        if request.error_context:
            error_section = f"""
## Previous Attempt Error
Your last attempt failed with the following error. Please fix this in your next version:
```
{request.error_context}
```
"""

        strategy_section = ""
        if request.strategy_hint:
            strategy_section = f"""
## Strategy Advisor Suggestion
Based on current hardware performance metrics, you should focus on the following:
{request.strategy_hint}
"""

        prompt = f"""You are an expert CUDA kernel optimizer.
{error_section}
{strategy_section}
## Task
Optimizing kernel: {kernel_name}
Description: {description}

## Current Kernel
```cuda
{request.kernel_code}
```

## Launch Configuration
Block: {request.meta.get('block')}, Grid: {request.meta.get('grid_str')}, SharedMem: {request.meta.get('shared_mem', 0)}

## Nsight Compute Optimization Opportunities
{request.optimization_text}

## Requirements
Write an optimized version of the kernel addressing ALL the NCU feedback above.
Signature MUST be: {signature}

Write ONLY a valid YAML file to {request.output_meta_path} with these fields:
  kernel_name: {kernel_name}
  kernel_code: |
    <optimized code>
  parameters:
{param_yaml_str}
  grid_str: <expression using parameters>
  block: [x, y, z]
  shared_mem: <integer bytes or string expression>

Do NOT explain. Just write the file.
"""

        prompt_file = self.scratch_dir / f"prompt_{request.version_label}.txt"
        out_file = self.scratch_dir / f"codex_out_{request.version_label}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        runtime_root = self._resolve_runtime_root()
        codex_sqlite_dir = runtime_root / "codex" / "sqlite"
        codex_logs_dir = runtime_root / "codex" / "logs"

        command = [
            *self.settings.codex_command,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--add-dir",
            str(self.settings.state_dir),
            "-c",
            f"sqlite_home={str(codex_sqlite_dir).__repr__()}",
            "-c",
            f"log_dir={str(codex_logs_dir).__repr__()}",
            "-C",
            str(self.settings.workspace),
            "-o",
            str(out_file),
            "-",
        ]
        codex_env = self._build_codex_env()

        print(f"  [codex] Generating {request.version_label}...")
        print(f"  [codex] Command: {' '.join(command)}")
        result = subprocess.run(
            command,
            stdin=prompt_file.open("r", encoding="utf-8"),
            capture_output=True,
            text=True,
            timeout=300,
            env=codex_env,
        )
        print(f"  [codex] exit={result.returncode}")
        if result.returncode != 0:
            print(f"  [codex] Prompt file: {prompt_file}")
            print(f"  [codex] Output file: {out_file}")
            print(f"  [codex] HOME={codex_env['HOME']}")
            print(f"  [codex] CODEX_HOME={codex_env['CODEX_HOME']}")
            print(f"  [codex] CODEX_SQLITE_HOME={codex_env['CODEX_SQLITE_HOME']}")
            if result.stdout:
                print(f"  [codex] stdout:\n{result.stdout}")
            if result.stderr:
                print(f"  [codex] stderr:\n{result.stderr}")

        if request.output_meta_path.exists():
            return True

        response = ""
        if out_file.exists():
            response = out_file.read_text(encoding="utf-8")
        elif result.stdout:
            response = result.stdout

        if response:
            parsed = _parse_yaml_response(response)
            if parsed:
                request.output_meta_path.parent.mkdir(parents=True, exist_ok=True)
                with request.output_meta_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(parsed, handle, sort_keys=False)
                return True

            print(f"  [codex] Could not parse YAML from response ({len(response)} chars)")
            print(f"  [codex] First 500 chars: {response[:500]}")

        return False


def _parse_yaml_response(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:yaml)?\s*\n", "", text, flags=re.M)
    text = re.sub(r"\n```\s*$", "", text, flags=re.M)
    match = re.search(r"(kernel_name:.*)", text, re.S)
    if match:
        text = match.group(1)
    try:
        meta = yaml.safe_load(text)
        if meta and "kernel_name" in meta and "kernel_code" in meta:
            return meta
    except yaml.YAMLError:
        pass
    return None


def create_backend(name: str, settings) -> KernelGenerationBackend:
    normalized = name.strip().lower()
    if normalized == "codex-exec":
        return CodexExecBackend(settings)
    raise RuntimeError(f"Unsupported kernel-generation backend: {name}")
