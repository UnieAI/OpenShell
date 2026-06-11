import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from tools.backends import create_backend
from tools.capabilities import (
    detect_runtime_capabilities,
    require_compile_capabilities,
    require_optimize_capabilities,
    require_profile_capabilities,
    require_verify_capabilities,
)
from tools.kernel_pipeline import compile_kernel, render_from_meta, verify_kernel
from tools.profilers import create_profile_executor
from tools.runtime import load_runtime_settings, resolve_user_path
import tools.search_optimizer as search_optimizer


def init_from_config(config_path: Path, settings):
    """Initialize the active kernel metadata from a kernel configuration YAML."""
    config = settings.load_yaml(config_path)
    baseline = config.get("baseline", {})
    meta = {
        "kernel_name": config.get("kernel_name"),
        "kernel_code": baseline.get("kernel_code"),
        "parameters": config.get("parameters"),
        "grid_str": baseline.get("grid_str"),
        "block": baseline.get("block"),
        "shared_mem": baseline.get("shared_mem", 0),
    }

    settings.active_kernel_yaml.parent.mkdir(parents=True, exist_ok=True)
    settings.dump_yaml(settings.active_kernel_yaml, meta)
    return config


def require_profile_executor(executor, action: str):
    reason = executor.describe_unavailability()
    if reason:
        raise RuntimeError(
            f"{action} requires a profiling executor, but '{executor.name}' is unavailable: {reason}"
        )


def require_mode_capabilities(args, capabilities) -> None:
    if args.init_baseline:
        return
    if args.optimize:
        if args.max_depth == 0:
            require_profile_capabilities(capabilities)
        else:
            require_optimize_capabilities(capabilities)
        return
    if args.profile and not args.test_only:
        require_profile_capabilities(capabilities)
        return
    if args.test_only:
        require_verify_capabilities(capabilities)
        return
    require_compile_capabilities(capabilities)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to kernel configuration YAML (for example configs/gemm.yaml)",
    )
    parser.add_argument(
        "--runtime-config",
        help="Path to the runtime config YAML. Defaults to AUTOAGENT_RUNTIME_CONFIG or <workspace>/config.yaml.",
    )
    parser.add_argument(
        "--workspace",
        help="Path to the source workspace that contains templates, configs, and Python modules.",
    )
    parser.add_argument(
        "--state-dir",
        help="Writable directory for generated kernels, build artifacts, and reports.",
    )
    parser.add_argument(
        "--backend",
        help="Kernel-generation backend override. Default comes from the runtime config.",
    )
    parser.add_argument(
        "--profile-executor",
        help="Profiling executor override. Use 'disabled' in standard OpenShell sandboxes.",
    )
    parser.add_argument("--profile", action="store_true", help="Run with profiling")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run verification and skip rendering plus compilation",
    )
    parser.add_argument(
        "--init-baseline",
        action="store_true",
        help="Only generate baseline metadata (active kernel YAML)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Trigger iterative tree-search optimization",
    )
    parser.add_argument(
        "--branch-factor",
        type=int,
        default=3,
        help="Number of candidate kernels to generate per depth when optimizing.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum search depth for optimization. Use 0 to profile only the baseline and still emit results.csv.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per branch when generation, rendering, compile, or verify fails.",
    )
    parser.add_argument(
        "--print-capabilities",
        action="store_true",
        help="Print detected runtime capabilities and exit.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    settings = load_runtime_settings(
        runtime_config_path=args.runtime_config,
        workspace=args.workspace,
        state_dir=args.state_dir,
    )
    if args.backend:
        settings = replace(settings, llm_backend=args.backend)
    if args.profile_executor:
        settings = replace(settings, profile_executor=args.profile_executor)

    start_time = time.time()
    capabilities = detect_runtime_capabilities(settings)

    if args.print_capabilities:
        print(f"Runtime capabilities: {capabilities.summary()}")
        sys.exit(0)

    problem_config_path = resolve_user_path(args.config, settings.workspace)
    if not problem_config_path.exists():
        print(f"Error: Configuration file {problem_config_path} not found.")
        sys.exit(1)

    try:
        require_mode_capabilities(args, capabilities)
        problem_config = init_from_config(problem_config_path, settings)
        kernel_name = problem_config.get("kernel_name", "kernel")

        if args.init_baseline:
            print(f"--- Initializing Baseline Meta from {problem_config_path} ---")
            print(f"Baseline meta generated in {settings.active_kernel_yaml}")
            print(f"\n--- Total orchestrator elapsed: {time.time() - start_time:.4f}s ---")
            sys.exit(0)

        if args.optimize:
            backend = (
                create_backend(settings.llm_backend, settings)
                if args.max_depth > 0
                else None
            )
            profiler_executor = create_profile_executor(settings.profile_executor, settings)
            require_profile_executor(profiler_executor, "Optimization")
            print(f"--- Triggering Tree-Search Optimization for {kernel_name} ---")
            search_optimizer.main(
                problem_config=problem_config_path,
                runtime_settings=settings,
                backend=backend,
                profiler_executor=profiler_executor,
                branch_factor=args.branch_factor,
                max_depth=args.max_depth,
                max_retries=args.max_retries,
            )
            print(f"\n--- Total orchestrator elapsed: {time.time() - start_time:.4f}s ---")
            sys.exit(0)

        if args.profile and not args.test_only:
            profiler_executor = create_profile_executor(settings.profile_executor, settings)
            require_profile_executor(profiler_executor, "Profiling")
            print(f"--- Starting Profiling for {kernel_name} ---")
            render_from_meta(
                meta_path=settings.active_kernel_yaml,
                template_path=settings.template_path,
                output_cu=settings.output_cu,
            )
            success, _ = compile_kernel(
                output_cu=settings.output_cu,
                output_so=settings.output_so,
                workspace_root=settings.workspace,
            )
            if not success:
                print(f"\n--- Total orchestrator elapsed: {time.time() - start_time:.4f}s ---")
                sys.exit(1)

            profiler_executor.profile(
                output_so=settings.output_so,
                meta_path=settings.active_kernel_yaml,
                report_path=settings.report_dir / "log",
                kernel_config_path=problem_config_path,
            )
        else:
            if not args.test_only:
                print(f"--- Meta-Driven Processing {kernel_name} ---")
                render_from_meta(
                    meta_path=settings.active_kernel_yaml,
                    template_path=settings.template_path,
                    output_cu=settings.output_cu,
                )
                success, _ = compile_kernel(
                    output_cu=settings.output_cu,
                    output_so=settings.output_so,
                    workspace_root=settings.workspace,
                )
                if not success:
                    print(f"\n--- Total orchestrator elapsed: {time.time() - start_time:.4f}s ---")
                    sys.exit(1)
            else:
                print(f"--- Running Verification Only for {kernel_name} ---")

            success, _ = verify_kernel(
                output_so=settings.output_so,
                meta_path=settings.active_kernel_yaml,
                config_path=problem_config_path,
                workspace_root=settings.workspace,
            )
            if success:
                print(f"{kernel_name.upper()} Verification Successful!")
            else:
                print(f"{kernel_name.upper()} Verification FAILED!")
                print(f"\n--- Total orchestrator elapsed: {time.time() - start_time:.4f}s ---")
                sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        print(f"Runtime capabilities: {capabilities.summary()}")
        sys.exit(1)

    print(f"\n--- Total orchestrator elapsed: {time.time() - start_time:.4f}s ---")
