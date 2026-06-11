#!/usr/bin/env python3
"""
Tree-search kernel optimizer.

At each depth, generates B candidate kernels via the configured LLM backend based
on NCU feedback, profiles each, picks the best, and branches from it at the next
depth.
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from tools.backends import GenerationRequest, create_backend
from tools.kernel_pipeline import compile_kernel, render_from_meta, verify_kernel
from tools.ncu_report_parser import NCUReportParser
from tools.profilers import create_profile_executor
from tools.runtime import load_runtime_settings

DEFAULT_BRANCH_FACTOR = 3
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_RETRIES = 3


def candidate_dir(kernels_dir: Path, depth: int, branch: int) -> Path:
    return kernels_dir / f"d{depth}" / f"b{branch}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def auto_suggest_strategy(metrics):
    """Analyze NCU metrics and return targeted optimization hints."""
    hints = []
    comp = metrics.get("compute_throughput_pct", 0)
    mem = metrics.get("memory_throughput_pct", 0)
    sm_busy = metrics.get("sm_busy_pct", 0)

    if mem > comp * 1.5:
        hints.append(
            "- The kernel is Memory-Bound. Implement **Double Buffering** (software pipelining) to overlap global memory loads with computation."
        )
        hints.append(
            "- Ensure you are using **Vectorized Loads** (e.g., float4) for all global memory accesses."
        )
    elif comp > 50 and sm_busy < 80:
        hints.append(
            "- The kernel is Compute-Bound but SMs are not fully utilized. Try **Increasing Work Per Thread** (Coarsening) to compute a larger sub-tile of C (e.g., 4x4 or 8x8) per thread."
        )

    if sm_busy < 20:
        hints.append(
            "- SM utilization is very low. Increase the **Block Size** or decrease resource usage (registers/shared mem) to improve occupancy."
        )

    if not hints:
        hints.append(
            "- Continue refining the current implementation, focusing on reducing instruction overhead and improving data reuse in shared memory."
        )

    return "\n".join(hints)


def record_result(results_csv: Path, label, metrics, desc):
    exists = results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "label",
                    "timestamp",
                    "description",
                    "duration_us",
                    "memory_throughput_pct",
                    "compute_throughput_pct",
                    "achieved_occupancy_pct",
                    "sm_busy_pct",
                    "memory_throughput_gbps",
                ]
            )
        writer.writerow(
            [
                label,
                datetime.now().isoformat(),
                desc,
                metrics.get("duration_us", ""),
                metrics.get("memory_throughput_pct", ""),
                metrics.get("compute_throughput_pct", ""),
                metrics.get("achieved_occupancy_pct", ""),
                metrics.get("sm_busy_pct", ""),
                metrics.get("memory_throughput_gbps", ""),
            ]
        )


def print_results(results_csv: Path):
    if not results_csv.exists():
        return
    with results_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return
    print("\n" + "=" * 120)
    print("TREE SEARCH RESULTS")
    print("=" * 120)
    print(
        f"{'Label':<10} {'Duration(us)':<14} {'MemThpt(%)':<12} {'CompThpt(%)':<13} "
        f"{'Occupancy(%)':<14} {'SMBusy(%)':<11} {'Mem(GB/s)':<11} {'Description'}"
    )
    print("-" * 120)
    for row in rows[1:]:
        print(
            f"{row[0]:<10} {row[3]:<14} {row[4]:<12} {row[5]:<13} "
            f"{row[6]:<14} {row[7]:<11} {row[8]:<11} {row[2][:40]}"
        )
    print("=" * 120)


def pick_best(kernels_dir: Path, depth: int, branch_factor: int):
    """Pick the branch with lowest duration_us at the given depth."""
    best_branch = None
    best_duration = float("inf")
    for branch in range(branch_factor):
        ncu_log = candidate_dir(kernels_dir, depth, branch) / "ncu_log.txt"
        if not ncu_log.exists():
            continue
        try:
            parser = NCUReportParser(ncu_log)
            duration = parser.metrics.get("duration_us", float("inf"))
            if duration < best_duration:
                best_duration = duration
                best_branch = branch
        except Exception:
            continue
    if best_branch is None:
        print(f"  [best] No successfully profiled candidates at depth {depth}")
        return None
    print(f"  [best] d{depth}/b{best_branch} (duration={best_duration}us)")
    return best_branch


def _validate_profile_executor(profiler_executor):
    reason = profiler_executor.describe_unavailability()
    if reason:
        raise RuntimeError(
            f"Tree-search optimization requires a working profile executor, but '{profiler_executor.name}' is unavailable: {reason}"
        )


def _validate_backend(backend):
    reason = backend.describe_unavailability()
    if reason:
        raise RuntimeError(
            f"Tree-search optimization requires a working generation backend, but '{backend.name}' is unavailable: {reason}"
        )


def main(
    problem_config=None,
    runtime_settings=None,
    backend=None,
    profiler_executor=None,
    branch_factor=None,
    max_depth=None,
    max_retries=None,
):
    runtime_settings = runtime_settings or load_runtime_settings()
    branch_factor = DEFAULT_BRANCH_FACTOR if branch_factor is None else branch_factor
    max_depth = DEFAULT_MAX_DEPTH if max_depth is None else max_depth
    max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries

    if max_depth < 0:
        raise RuntimeError("max_depth must be >= 0")
    if branch_factor < 1:
        raise RuntimeError("branch_factor must be >= 1")
    if max_retries < 1:
        raise RuntimeError("max_retries must be >= 1")

    backend = backend or (
        create_backend(runtime_settings.llm_backend, runtime_settings) if max_depth > 0 else None
    )
    profiler_executor = profiler_executor or create_profile_executor(
        runtime_settings.profile_executor, runtime_settings
    )
    _validate_profile_executor(profiler_executor)
    if backend is not None:
        _validate_backend(backend)

    problem_config = Path(problem_config or runtime_settings.active_kernel_yaml).resolve()
    kernels_dir = runtime_settings.kernels_dir
    results_csv = kernels_dir / "results.csv"

    print("=" * 60)
    print("Tree Search Kernel Optimizer")
    print(f"  Problem Config: {problem_config}")
    print(f"  Branching factor: {branch_factor}")
    print(f"  Max depth: {max_depth}")
    print(f"  Backend: {backend.name if backend is not None else 'none (baseline-only profiling)'}")
    print(f"  Profile executor: {profiler_executor.name}")
    print("=" * 60)

    if kernels_dir.exists():
        shutil.rmtree(kernels_dir)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Depth 0: Baseline ---")
    d0 = candidate_dir(kernels_dir, 0, 0)
    ensure_dir(d0)

    active_kernel_meta = runtime_settings.active_kernel_yaml
    d0_meta = d0 / "active_kernel.yaml"
    d0_cu = d0 / "kernel.cu"
    d0_so = d0 / "kernel.so"
    d0_ncu = d0 / "ncu_log.txt"

    if active_kernel_meta.exists():
        shutil.copy(active_kernel_meta, d0_meta)

    render_from_meta(d0_meta, runtime_settings.template_path, d0_cu)
    success, err = compile_kernel(d0_cu, d0_so, workspace_root=runtime_settings.workspace)
    if not success:
        print(f"Baseline compile failed: {err}")
        return
    success, err = verify_kernel(
        d0_so,
        d0_meta,
        config_path=problem_config,
        workspace_root=runtime_settings.workspace,
    )
    if not success:
        print(f"Baseline verification failed: {err}")
        return
    if not profiler_executor.profile(d0_so, d0_meta, d0_ncu, problem_config):
        print("Baseline profiling failed!")
        return

    parser = NCUReportParser(d0_ncu)
    record_result(results_csv, "d0/b0", parser.metrics, "baseline")
    print(f"  Baseline: duration={parser.metrics.get('duration_us')}us")

    parent_dir = d0
    for depth in range(1, max_depth + 1):
        print(f"\n{'=' * 60}")
        print(f"--- Depth {depth}: Generating {branch_factor} candidates ---")
        print(f"{'=' * 60}")

        parent_meta_path = parent_dir / "active_kernel.yaml"
        parent_ncu_path = parent_dir / "ncu_log.txt"

        with parent_meta_path.open("r", encoding="utf-8") as handle:
            parent_meta = yaml.safe_load(handle) or {}
        parent_kernel = parent_meta["kernel_code"]

        parent_parser = NCUReportParser(parent_ncu_path)
        opt_text = parent_parser.opt_text()
        strategy_hint = auto_suggest_strategy(parent_parser.metrics)

        print(f"  Parent: {parent_dir}")
        print(f"  Parent duration: {parent_parser.metrics.get('duration_us')}us")

        for branch in range(branch_factor):
            print(f"\n  --- d{depth}/b{branch} ---")
            cdir = candidate_dir(kernels_dir, depth, branch)
            ensure_dir(cdir)

            c_meta = cdir / "active_kernel.yaml"
            c_cu = cdir / "kernel.cu"
            c_so = cdir / "kernel.so"
            c_ncu = cdir / "ncu_log.txt"

            label = f"d{depth}_b{branch}"

            error_msg = None
            branch_success = False
            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"  [retry] Attempt {attempt + 1}/{max_retries} for {label}...")

                request = GenerationRequest(
                    kernel_code=parent_kernel,
                    optimization_text=opt_text,
                    meta=parent_meta,
                    version_label=label,
                    output_meta_path=c_meta,
                    kernel_config_path=problem_config,
                    error_context=error_msg,
                    strategy_hint=strategy_hint,
                )
                if backend is None or not backend.generate(request):
                    error_msg = "The configured LLM backend failed to generate a YAML response."
                    continue

                try:
                    render_from_meta(c_meta, runtime_settings.template_path, c_cu)
                except Exception as exc:
                    error_msg = f"Render failed: {exc}"
                    continue

                success, err = compile_kernel(c_cu, c_so, workspace_root=runtime_settings.workspace)
                if not success:
                    error_msg = err
                    continue

                success, err = verify_kernel(
                    c_so,
                    c_meta,
                    config_path=problem_config,
                    workspace_root=runtime_settings.workspace,
                )
                if not success:
                    error_msg = err
                    continue

                branch_success = True
                break

            if not branch_success:
                print(f"  [skip] Exhausted retries for {label}. Last error: {error_msg}")
                continue

            if not profiler_executor.profile(c_so, c_meta, c_ncu, problem_config):
                print(f"  [skip] Profile failed for {label}")
                continue

            try:
                candidate_parser = NCUReportParser(c_ncu)
                record_result(
                    results_csv,
                    f"d{depth}/b{branch}",
                    candidate_parser.metrics,
                    f"{backend.name} d{depth} branch {branch}",
                )
                print(f"  duration={candidate_parser.metrics.get('duration_us')}us")
            except Exception as exc:
                print(f"  [skip] Failed to parse NCU log: {exc}")

        best_branch = pick_best(kernels_dir, depth, branch_factor)
        if best_branch is None:
            print(f"Stopping search at depth {depth}: no candidate produced a parsable NCU result.")
            break
        parent_dir = candidate_dir(kernels_dir, depth, best_branch)

    print_results(results_csv)

    ensure_dir(runtime_settings.output_dir)

    best_meta_src = parent_dir / "active_kernel.yaml"
    best_cu_src = parent_dir / "kernel.cu"
    best_so_src = parent_dir / "kernel.so"

    if best_meta_src.exists():
        shutil.copy(best_meta_src, runtime_settings.output_dir / "best_kernel.yaml")
    if best_cu_src.exists():
        shutil.copy(best_cu_src, runtime_settings.output_dir / "best_kernel.cu")
    if best_so_src.exists():
        shutil.copy(best_so_src, runtime_settings.output_dir / "best_kernel.so")

    print(f"\nBest kernel artifacts saved to {runtime_settings.output_dir}/")


if __name__ == "__main__":
    main()
