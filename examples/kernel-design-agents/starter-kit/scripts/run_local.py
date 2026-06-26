"""
FlashInfer-Bench local benchmark runner with staged KDA-friendly controls.

It packs the current solution, optionally restricts the workload set, lowers the
benchmark budget for execute-mode smoke runs, and can tee stdout/stderr into a
workspace artifact file.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution

LOGGER = logging.getLogger("kda.run_local")


class TeeStream:
    """Mirror writes to multiple file-like streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def tee_output(log_path):
    """Duplicate stdout/stderr to a log file when requested."""
    if not log_path:
        yield
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open("w", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def parse_optional_int(value, field_name: str):
    """Parse positive integers from CLI/env values."""
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def parse_uuid_list(value) -> list[str]:
    """Parse a comma-separated workload UUID list."""
    if value in (None, ""):
        return []
    items = [item.strip() for item in value.replace("\n", ",").split(",")]
    return [item for item in items if item]


def get_error_detail(evaluation):
    """Best-effort extraction of actionable evaluation failure details."""
    candidate_attrs = (
        "message",
        "error",
        "error_message",
        "compile_error",
        "builder_error",
        "stderr",
        "stdout",
        "details",
    )

    for attr in candidate_attrs:
        value = getattr(evaluation, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_dump = getattr(evaluation, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        nested = get_error_detail_from_mapping(dumped)
        if nested:
            return nested

    return None


def get_error_detail_from_mapping(value):
    """Recursively search mappings/lists for likely error strings."""
    interesting_keys = (
        "message",
        "error",
        "error_message",
        "stderr",
        "stdout",
        "detail",
        "details",
        "traceback",
        "exception",
        "reason",
    )

    if isinstance(value, dict):
        for key in interesting_keys:
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        for nested in value.values():
            detail = get_error_detail_from_mapping(nested)
            if detail:
                return detail

    if isinstance(value, list):
        for nested in value:
            detail = get_error_detail_from_mapping(nested)
            if detail:
                return detail

    return None


def configure_logging():
    """Ensure flashinfer-bench logger output reaches the workspace log."""
    level_name = os.environ.get("KDA_BENCHMARK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )
    LOGGER.debug("Configured logging with level=%s", level_name)


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def extract_workload_uuid(workload_like) -> str:
    """Read the workload UUID from either direct or nested workload objects."""
    direct_uuid = getattr(workload_like, "uuid", None)
    if isinstance(direct_uuid, str) and direct_uuid:
        return direct_uuid

    nested_workload = getattr(workload_like, "workload", None)
    nested_uuid = getattr(nested_workload, "uuid", None)
    if isinstance(nested_uuid, str) and nested_uuid:
        return nested_uuid

    raise AttributeError(f"Unable to resolve workload UUID from object: {workload_like!r}")


def resolve_dump_traces(requested: bool, trace_set_path: str) -> bool:
    """Disable trace dumping automatically when the dataset mount is read-only."""
    if not requested:
        return False

    if os.access(trace_set_path, os.W_OK):
        return True

    print(
        "Disabling trace dumping because the trace-set path is not writable: "
        f"{trace_set_path}",
        file=sys.stderr,
    )
    return False


def select_workloads(all_workloads, workload_limit, workload_uuids: list[str]):
    """Choose either an explicit subset of workloads or the first N entries."""
    if workload_uuids:
        workload_by_uuid = {extract_workload_uuid(workload): workload for workload in all_workloads}
        missing = [uuid for uuid in workload_uuids if uuid not in workload_by_uuid]
        if missing:
            raise ValueError(f"Requested workload UUIDs not found: {missing}")
        return [workload_by_uuid[uuid] for uuid in workload_uuids]

    if workload_limit is not None:
        return list(all_workloads[:workload_limit])

    return list(all_workloads)


def build_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Construct BenchmarkConfig from staged-run settings."""
    return BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
    )


def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig,
    workload_limit,
    workload_uuids: list[str],
    dump_traces: bool,
) -> tuple[dict, list]:
    """Run benchmark locally and return both results and selected workloads."""
    trace_set_path = get_trace_set_path()
    dump_traces = resolve_dump_traces(dump_traces, trace_set_path)
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    all_workloads = trace_set.workloads.get(solution.definition, [])
    selected_workloads = select_workloads(all_workloads, workload_limit, workload_uuids)

    if not selected_workloads:
        raise ValueError(f"No workloads selected for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: selected_workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=dump_traces)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            error_detail = get_error_detail(trace.evaluation)
            if error_detail:
                entry["error_detail"] = error_detail
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][extract_workload_uuid(trace.workload)] = entry

    return results, selected_workloads


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            if result.get("error_detail"):
                detail = result["error_detail"].replace("\n", " ").strip()
                if len(detail) > 200:
                    detail = detail[:197] + "..."
                print(f" | detail={detail}", end="")

            print()


def write_results_json(
    output_path,
    solution: Solution,
    config: BenchmarkConfig,
    selected_workloads: list,
    results: dict,
):
    """Persist a structured benchmark artifact when requested."""
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "solution": {
            "name": solution.name,
            "definition": solution.definition,
            "author": solution.author,
        },
        "benchmark_config": {
            "warmup_runs": config.warmup_runs,
            "iterations": config.iterations,
            "num_trials": config.num_trials,
        },
        "selected_workloads": [extract_workload_uuid(workload) for workload in selected_workloads],
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """CLI for local benchmark staging."""
    parser = argparse.ArgumentParser(description="Pack solution and run local FlashInfer benchmarks")
    parser.add_argument("--warmup-runs", type=int, default=parse_optional_int(os.environ.get("KDA_BENCHMARK_WARMUP_RUNS"), "warmup_runs") or 3)
    parser.add_argument("--iterations", type=int, default=parse_optional_int(os.environ.get("KDA_BENCHMARK_ITERATIONS"), "iterations") or 100)
    parser.add_argument("--num-trials", type=int, default=parse_optional_int(os.environ.get("KDA_BENCHMARK_NUM_TRIALS"), "num_trials") or 5)
    parser.add_argument("--workload-limit", type=int, default=parse_optional_int(os.environ.get("KDA_BENCHMARK_WORKLOAD_LIMIT"), "workload_limit"))
    parser.add_argument("--workload-uuids", default=os.environ.get("KDA_BENCHMARK_WORKLOAD_UUIDS", ""))
    parser.add_argument("--log-file", default=os.environ.get("KDA_RUN_LOCAL_LOG_PATH", ""))
    parser.add_argument("--results-json", default=os.environ.get("KDA_RUN_LOCAL_RESULTS_PATH", ""))
    parser.add_argument("--no-dump-traces", action="store_true")
    return parser


def main():
    """Pack solution and run benchmark."""
    args = build_parser().parse_args()
    workload_uuids = parse_uuid_list(args.workload_uuids)
    dump_traces = not args.no_dump_traces

    with tee_output(args.log_file or None):
        try:
            configure_logging()
            print("Packing solution from source files...")
            solution_path = pack_solution()

            print("\nLoading solution...")
            solution = Solution.model_validate_json(solution_path.read_text())
            print(f"Loaded: {solution.name} ({solution.definition})")

            config = build_benchmark_config(args)
            print("\nBenchmark settings:")
            print(f"  warmup_runs={config.warmup_runs}")
            print(f"  iterations={config.iterations}")
            print(f"  num_trials={config.num_trials}")
            if workload_uuids:
                print(f"  workload_uuids={','.join(workload_uuids)}")
            elif args.workload_limit is not None:
                print(f"  workload_limit={args.workload_limit}")
            else:
                print("  workloads=all")

            print("\nRunning benchmark...")
            results, selected_workloads = run_benchmark(
                solution=solution,
                config=config,
                workload_limit=args.workload_limit,
                workload_uuids=workload_uuids,
                dump_traces=dump_traces,
            )

            print(f"Selected workloads: {len(selected_workloads)}")
            if selected_workloads:
                print("  " + ", ".join(extract_workload_uuid(workload) for workload in selected_workloads))

            if not results:
                print("No results returned!")
                return

            print_results(results)
            write_results_json(args.results_json or None, solution, config, selected_workloads, results)
        except Exception:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
