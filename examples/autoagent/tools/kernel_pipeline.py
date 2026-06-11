import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Template


def render_from_meta(meta_path=None, template_path=None, output_cu=None):
    """Render a CUDA source file from kernel metadata."""
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle)

    with open(template_path, "r", encoding="utf-8") as handle:
        template = Template(handle.read())

    source = template.render(**meta)
    output_path = Path(output_cu)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    return True


def _candidate_tvm_roots(workspace_root: Path) -> list[Path]:
    roots: list[Path] = []

    try:
        import tvm_ffi

        roots.append(Path(tvm_ffi.__file__).resolve().parent)
    except Exception:
        pass

    interpreter_root = Path(sys.executable).resolve().parent.parent
    roots.extend(
        [
            interpreter_root / "lib/python3.12/site-packages/tvm_ffi",
            Path("/opt/autoagent-venv/lib/python3.12/site-packages/tvm_ffi"),
            workspace_root / ".venv/lib/python3.12/site-packages/tvm_ffi",
        ]
    )

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique_roots.append(root)
    return unique_roots


def _resolve_tvm_ffi_paths(workspace_root: Path) -> tuple[str, str]:
    for root in _candidate_tvm_roots(workspace_root):
        include_dir = root / "include"
        lib_dir = root / "lib"
        if include_dir.exists() and lib_dir.exists():
            return str(include_dir), str(lib_dir)

    raise RuntimeError(
        "Unable to locate tvm_ffi headers and libraries. "
        "Expected an installed tvm_ffi package under the runtime environment."
    )


def _resolve_nvcc() -> str:
    candidates = [
        shutil.which("nvcc"),
        os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise RuntimeError(
        "nvcc not found. Ensure the image provides the CUDA toolkit and that "
        "PATH or CUDA_HOME points to the toolkit bin directory."
    )


def compile_kernel(output_cu=None, output_so=None, include_paths=None, lib_paths=None, workspace_root=None):
    """Compile a CUDA kernel into a shared object."""
    output_cu = Path(output_cu)
    output_so = Path(output_so)
    workspace_root = Path(workspace_root or Path.cwd())
    default_include, default_lib = _resolve_tvm_ffi_paths(workspace_root)
    nvcc = _resolve_nvcc()

    include_paths = include_paths or [default_include, "/usr/local/cuda/include"]
    lib_paths = lib_paths or [default_lib]

    output_so.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nvcc, "-shared", "-Xcompiler", "-fPIC", str(output_cu), "-o", str(output_so)]
    for path in include_paths:
        cmd.append(f"-I{path}")
    for path in lib_paths:
        cmd.append(f"-L{path}")
    cmd.append("-ltvm_ffi")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = f"Compilation failed: {result.stderr}"
        print(err)
        return False, err
    return True, ""


def _generate_test_script(output_so, meta_path, config_path, workspace_root, is_verify=True):
    """Generate a Python script for verification or profiling based on config."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    test_inputs = config.get("test_inputs", [])
    verification = config.get("verification", {})
    ref_code = verification.get("reference_code", "")
    tolerance = verification.get("tolerance", {"atol": 1e-3, "rtol": 1e-3})

    input_gen_lines = []
    arg_names = []
    for input_spec in test_inputs:
        name = input_spec["name"]
        arg_names.append(name)
        if input_spec["type"] == "tensor":
            shape = input_spec["shape"]
            generator = input_spec.get("generator", "randn")
            dtype = input_spec.get("dtype", "float32")
            if generator == "randn":
                input_gen_lines.append(
                    f"    {name} = torch.randn({shape}, device='cuda', dtype=torch.{dtype})"
                )
            elif generator == "zeros":
                input_gen_lines.append(
                    f"    {name} = torch.zeros({shape}, device='cuda', dtype=torch.{dtype})"
                )
            elif generator == "ones":
                input_gen_lines.append(
                    f"    {name} = torch.ones({shape}, device='cuda', dtype=torch.{dtype})"
                )
        else:
            value = input_spec.get("value")
            input_gen_lines.append(f"    {name} = {value}")

    args_str = ", ".join(arg_names)
    script = f"""
import sys
import torch

sys.path.insert(0, {str(Path(workspace_root).resolve())!r})
from tools.cuda_dispatcher import CUDADispatcher as UniversalDispatcher

def run():
{chr(10).join(input_gen_lines)}

    dispatcher = UniversalDispatcher({str(Path(output_so).resolve())!r}, {str(Path(meta_path).resolve())!r})
    dispatcher({args_str})
"""
    if is_verify:
        ref_code_indented = "\n".join(["    " + line for line in ref_code.splitlines()])
        script += f"""
    # Reference calculation
{ref_code_indented}

    torch.testing.assert_close(
        output if 'output' in locals() else C if 'C' in locals() else expected,
        expected,
        atol={tolerance.get('atol')},
        rtol={tolerance.get('rtol')},
    )
    print("PASS")
"""

    script += """
if __name__ == "__main__":
    run()
"""
    return script


def verify_kernel(output_so=None, meta_path=None, config_path=None, workspace_root=None):
    """Verify kernel correctness using a subprocess and dynamic script."""
    if not config_path:
        return False, "No config_path provided for verification"

    script = _generate_test_script(
        output_so=output_so,
        meta_path=meta_path,
        config_path=config_path,
        workspace_root=workspace_root or Path.cwd(),
        is_verify=True,
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if result.returncode == 0 and "PASS" in result.stdout:
        return True, ""

    err = f"Verification failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    print(err)
    return False, err


def write_profile_script(output_so, meta_path, config_path, workspace_root, script_path):
    """Materialize the Python runner used by external profiling executors."""
    script = _generate_test_script(
        output_so=output_so,
        meta_path=meta_path,
        config_path=config_path,
        workspace_root=workspace_root,
        is_verify=False,
    )
    script_path = Path(script_path)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle) or {}
    return meta.get("kernel_name")
