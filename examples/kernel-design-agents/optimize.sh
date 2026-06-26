#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="${KDA_EXAMPLE_DIR:-${ROOT}/examples/kernel-design-agents}"
CONFIG_PATH="${KDA_CONFIG:-${EXAMPLE_DIR}/config/kda-gemm-task.yml}"
WORKSPACE="${KDA_WORKSPACE:-${EXAMPLE_DIR}/results/kda-task}"
STARTER_KIT_DIR="${KDA_STARTER_KIT_DIR:-${EXAMPLE_DIR}/starter-kit}"
GPU_SPEC="${KDA_GPU_SPEC:-1}"
MODEL="${KDA_CODEX_MODEL:-gpt-5.4-mini}"
REASONING="${KDA_CODEX_REASONING:-low}"
IMAGE="${KDA_IMAGE:-openshell-kda-example}"
MODE="${KDA_MODE:-draft}"
DOCKER_USER_DEFAULT="$(id -u):$(id -g)"
DOCKER_USER="${KDA_DOCKER_USER:-${DOCKER_USER_DEFAULT}}"
HOST_RUNTIME_ROOT="${KDA_HOST_RUNTIME_ROOT:-${TMPDIR:-/tmp}/openshell-kda-runtime}"
FIB_DATASET_PATH_VALUE="${KDA_FIB_DATASET_PATH:-${FIB_DATASET_PATH:-}}"
CONTAINER_FIB_DATASET_PATH="${KDA_CONTAINER_FIB_DATASET_PATH:-/datasets/mlsys26-contest}"
AUTO_SCAFFOLD=1
FORCE_SCAFFOLD=0
SCAFFOLDED_WORKSPACE=0
BUILD_IMAGE=0
KERNEL_OPTIMIZE=0
MAX_DEPTH="${KDA_MAX_DEPTH:-1}"
BRANCH="${KDA_BRANCH:-1}"

SOLUTION_NAME="${KDA_SOLUTION_NAME:-}"
DEFINITION="${KDA_DEFINITION:-}"
AUTHOR="${KDA_AUTHOR:-}"
LANGUAGE="${KDA_LANGUAGE:-}"
ENTRY_POINT="${KDA_ENTRY_POINT:-}"
SOURCE_DIR="${KDA_SOURCE_DIR:-}"
DESTINATION_PASSING_STYLE="${KDA_DESTINATION_PASSING_STYLE:-}"
BINDING="${KDA_BINDING:-}"
BENCHMARK_WARMUP_RUNS="${KDA_BENCHMARK_WARMUP_RUNS:-}"
BENCHMARK_ITERATIONS="${KDA_BENCHMARK_ITERATIONS:-}"
BENCHMARK_NUM_TRIALS="${KDA_BENCHMARK_NUM_TRIALS:-}"
BENCHMARK_WORKLOAD_LIMIT="${KDA_BENCHMARK_WORKLOAD_LIMIT:-}"
BENCHMARK_WORKLOAD_UUIDS="${KDA_BENCHMARK_WORKLOAD_UUIDS:-}"
PRESET_NAME="${KDA_PRESET_NAME:-}"
TASK_FAMILY="${KDA_TASK_FAMILY:-}"
WORKLOAD_PROFILE="${KDA_WORKLOAD_PROFILE:-}"

TASK_NAME="${KDA_TASK_NAME:-}"
OBJECTIVE="${KDA_OBJECTIVE:-}"
CORRECTNESS="${KDA_CORRECTNESS:-}"
TARGET="${KDA_TARGET:-}"
ALLOWED="${KDA_ALLOWED:-}"
VALIDATE="${KDA_VALIDATE:-}"
EVALUATE="${KDA_EVALUATE:-}"
PROMOTE="${KDA_PROMOTE:-}"

OPENAI_API_KEY_VALUE="${OPENAI_API_KEY:-}"
CODEX_API_KEY_VALUE="${CODEX_API_KEY:-}"
AUTH_MODE="none"

usage() {
    cat <<'EOF'
Usage: bash examples/kernel-design-agents/optimize.sh [options]

Single-entry Docker runner for the OpenShell KDA example. It scaffolds a task
workspace from the vendored FlashInfer starter kit when needed, resolves the
task contract and `config.toml` from config, then bind-mounts that workspace
into the example image and runs one of three flows directly against host files:
draft-only, single baseline smoke execute, or execute plus kernel optimization.

Options:
  --config=<path>         YAML config file. Default: examples/kernel-design-agents/config/kda-gemm-task.yml
  --workspace=<path>      Local task workspace.
  --starter-kit=<path>    Starter-kit template source. Default: examples/kernel-design-agents/starter-kit
  --gpus=<spec>           Docker GPU selector. Examples: 1, all, device=1
  --model=<name>          Codex model override.
  --reasoning=<level>     Codex reasoning effort. Default: low
  --image=<tag>           Local Docker image tag. Default: openshell-kda-example
  --mode=<name>           `draft` or `execute`. Default: draft
  --execute               Shorthand for `--mode=execute`
  --docker-user=<u:g>     Container user. Default: current host uid:gid
  --host-runtime-root=<p> Host directory for temporary Codex runtime state
  --fib-dataset-path=<p>  Host path to the FlashInfer Trace dataset. Mounted
                          read-only into the container and exported as
                          FIB_DATASET_PATH for execution mode.
  --build                 Build the example image before running.
  --kernel-optimize       Enable the optimization flow. Without this flag,
                          `--mode=execute` only runs one baseline smoke /
                          baseline revalidation attempt. With this flag,
                          use the current source candidate if one exists;
                          otherwise bootstrap a baseline first, then enter
                          the depth/branch optimization loop.
  --max-depth=<n>         Maximum number of promoted depth levels to pursue
                          in this run. Default: 1.
  --branch=<n>            Maximum number of new attempts per baseline/depth
                          level in this run. Default: 1.
  --no-scaffold           Require an existing workspace and TASK_CONTRACT.md.
  --clean-workspace       Recreate scaffolded files before the run.
  --force-scaffold        Backward-compatible alias for --clean-workspace.
  --solution-name=<text>  Write workspace config.toml automatically.
  --definition=<text>     Write workspace config.toml automatically.
  --author=<text>         Write workspace config.toml automatically.
  --language=<text>       Write workspace config.toml automatically.
  --entry-point=<text>    Write workspace config.toml automatically.
  --source-dir=<text>     Write workspace config.toml automatically.
  --destination-passing-style=<bool>
                          Write workspace config.toml automatically.
  --binding=<text>        Write workspace config.toml automatically.
  --benchmark-warmup-runs=<n>
                          Export reduced benchmark settings into the container.
  --benchmark-iterations=<n>
                          Export reduced benchmark settings into the container.
  --benchmark-num-trials=<n>
                          Export reduced benchmark settings into the container.
  --benchmark-workload-limit=<n>
                          Limit run_local.py to the first N workloads.
  --benchmark-workload-uuids=<list>
                          Comma-separated workload UUID subset for run_local.py.
  --task-name=<text>      Fill TASK_CONTRACT.md automatically.
  --objective=<text>      Fill TASK_CONTRACT.md automatically.
  --correctness=<text>    Fill TASK_CONTRACT.md automatically.
  --target=<text>         Fill TASK_CONTRACT.md automatically.
  --allowed=<text>        Fill TASK_CONTRACT.md automatically.
  --validate=<cmd>        Fill TASK_CONTRACT.md automatically.
  --evaluate=<cmd>        Fill TASK_CONTRACT.md automatically.
  --promote=<text>        Fill TASK_CONTRACT.md automatically.
  -h, --help              Show this help.

Auth:
  Prefer OPENAI_API_KEY or CODEX_API_KEY.
  If neither is set, the script falls back to ~/.codex/auth.json.

Examples:
  OPENAI_API_KEY='sk-...' \
  bash examples/kernel-design-agents/optimize.sh \
    --config=examples/kernel-design-agents/config/kda-gemm-task.yml \
    --gpus=1
EOF
}

trim_value() {
    local value="$1"
    value="$(printf '%s' "${value}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
        value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
        value="${value:1:${#value}-2}"
    fi
    printf '%s\n' "${value}"
}

to_abs_path() {
    local value="$1"
    local normalized="$1"
    local root_name="${ROOT##*/}"
    if [[ "${value}" == /* ]]; then
        printf '%s\n' "${value}"
        return 0
    fi
    normalized="${normalized#./}"
    if [[ "${normalized}" == "${root_name}/"* ]]; then
        normalized="${normalized#${root_name}/}"
    fi
    printf '%s/%s\n' "${ROOT}" "${normalized}"
}

normalize_mode() {
    local value="$1"
    case "${value}" in
        draft|execute)
            printf '%s\n' "${value}"
            ;;
        *)
            echo "Unsupported mode: ${value}. Expected draft or execute." >&2
            exit 2
            ;;
    esac
}

normalize_docker_gpus() {
    local value="$1"
    if [[ "${value}" == nvidia.com/gpu=* ]]; then
        printf 'device=%s\n' "${value#nvidia.com/gpu=}"
        return 0
    fi
    if [[ "${value}" == "all" || "${value}" == *"="* ]]; then
        printf '%s\n' "${value}"
        return 0
    fi
    printf 'device=%s\n' "${value}"
}

read_workspace_definition() {
    local config_path="$1"

    [[ -f "${config_path}" ]] || return 0

    python3 - "${config_path}" <<'PY'
import pathlib
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib

path = pathlib.Path(sys.argv[1])
try:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

definition = data.get("solution", {}).get("definition", "")
if isinstance(definition, str) and definition.strip():
    print(definition.strip())
PY
}

load_simple_yaml_config() {
    local path="$1"
    local line key value

    [[ -f "${path}" ]] || return 0

    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%%#*}"
        line="$(trim_value "${line}")"
        [[ -z "${line}" ]] && continue
        [[ "${line}" == *:* ]] || continue

        key="$(trim_value "${line%%:*}")"
        value="$(trim_value "${line#*:}")"

        case "${key}" in
            workspace) WORKSPACE="${value}" ;;
            starter_kit) STARTER_KIT_DIR="${value}" ;;
            gpus) GPU_SPEC="${value}" ;;
            model) MODEL="${value}" ;;
            reasoning) REASONING="${value}" ;;
            image) IMAGE="${value}" ;;
            mode) MODE="${value}" ;;
            docker_user) DOCKER_USER="${value}" ;;
            host_runtime_root) HOST_RUNTIME_ROOT="${value}" ;;
            fib_dataset_path) FIB_DATASET_PATH_VALUE="${value}" ;;
            auto_scaffold) [[ "${value}" == "false" ]] && AUTO_SCAFFOLD=0 || AUTO_SCAFFOLD=1 ;;
            force_scaffold) [[ "${value}" == "true" ]] && FORCE_SCAFFOLD=1 || FORCE_SCAFFOLD=0 ;;
            build_image) [[ "${value}" == "true" ]] && BUILD_IMAGE=1 || BUILD_IMAGE=0 ;;
            kernel_optimize) [[ "${value}" == "true" ]] && KERNEL_OPTIMIZE=1 || KERNEL_OPTIMIZE=0 ;;
            max_depth) MAX_DEPTH="${value}" ;;
            branch) BRANCH="${value}" ;;
            solution_name) SOLUTION_NAME="${value}" ;;
            definition) DEFINITION="${value}" ;;
            author) AUTHOR="${value}" ;;
            language) LANGUAGE="${value}" ;;
            entry_point) ENTRY_POINT="${value}" ;;
            source_dir) SOURCE_DIR="${value}" ;;
            destination_passing_style) DESTINATION_PASSING_STYLE="${value}" ;;
            binding) BINDING="${value}" ;;
            benchmark_warmup_runs) BENCHMARK_WARMUP_RUNS="${value}" ;;
            benchmark_iterations) BENCHMARK_ITERATIONS="${value}" ;;
            benchmark_num_trials) BENCHMARK_NUM_TRIALS="${value}" ;;
            benchmark_workload_limit) BENCHMARK_WORKLOAD_LIMIT="${value}" ;;
            benchmark_workload_uuids) BENCHMARK_WORKLOAD_UUIDS="${value}" ;;
            preset_name) PRESET_NAME="${value}" ;;
            task_family) TASK_FAMILY="${value}" ;;
            workload_profile) WORKLOAD_PROFILE="${value}" ;;
            task_name) TASK_NAME="${value}" ;;
            objective) OBJECTIVE="${value}" ;;
            correctness) CORRECTNESS="${value}" ;;
            target) TARGET="${value}" ;;
            allowed) ALLOWED="${value}" ;;
            validate) VALIDATE="${value}" ;;
            evaluate) EVALUATE="${value}" ;;
            promote) PROMOTE="${value}" ;;
        esac
    done < "${path}"
}

write_contract_if_requested() {
    local contract_path="$1"

    if [[ -z "${TASK_NAME}" && -z "${OBJECTIVE}" && -z "${CORRECTNESS}" && -z "${TARGET}" && -z "${ALLOWED}" && -z "${VALIDATE}" && -z "${EVALUATE}" && -z "${PROMOTE}" ]]; then
        return 0
    fi

    cat > "${contract_path}" <<EOF
# Task Contract

- Task name: ${TASK_NAME:-<fill in>}
- Objective: ${OBJECTIVE:-<fill in the user-facing goal>}
- Correctness requirements: ${CORRECTNESS:-<fill in required behavior, tolerances, or invariants>}
- Performance or quality target: ${TARGET:-<fill in measurable target if any>}
- Allowed implementation approaches: ${ALLOWED:-<fill in languages, libraries, APIs, or constraints>}
- Validation command: ${VALIDATE:-<fill in the command that proves correctness>}
- Evaluation command: ${EVALUATE:-<fill in the command that measures the target, if different>}
- Promotion criteria: ${PROMOTE:-<fill in what must be true before a candidate is accepted>}
EOF
}

write_starter_config_if_requested() {
    local config_path="$1"
    local source_dir_line=""
    local dps_line=""
    local binding_line=""
    local normalized_entry_point="${ENTRY_POINT:-kernel}"

    if [[ -z "${SOLUTION_NAME}" && -z "${DEFINITION}" && -z "${AUTHOR}" && -z "${LANGUAGE}" && -z "${ENTRY_POINT}" && -z "${SOURCE_DIR}" && -z "${DESTINATION_PASSING_STYLE}" && -z "${BINDING}" ]]; then
        return 0
    fi

    if [[ -n "${SOURCE_DIR}" ]]; then
        source_dir_line="source_dir = \"${SOURCE_DIR}\""
    fi
    if [[ -n "${DESTINATION_PASSING_STYLE}" ]]; then
        case "${DESTINATION_PASSING_STYLE}" in
            true|false)
                dps_line="destination_passing_style = ${DESTINATION_PASSING_STYLE}"
                ;;
            *)
                echo "Unsupported destination_passing_style: ${DESTINATION_PASSING_STYLE}. Expected true or false." >&2
                exit 2
                ;;
        esac
    fi
    if [[ -n "${BINDING}" ]]; then
        case "${BINDING}" in
            tvm-ffi|torch)
                binding_line="binding = \"${BINDING}\""
                ;;
            *)
                echo "Unsupported binding: ${BINDING}. Expected tvm-ffi or torch." >&2
                exit 2
                ;;
        esac
    fi

    if [[ "${LANGUAGE:-}" == "cuda" && "${BINDING:-}" == "torch" ]]; then
        case "${normalized_entry_point}" in
            ""|kernel)
                normalized_entry_point="kernel.cu::kernel"
                ;;
        esac
    fi

    cat > "${config_path}" <<EOF
[solution]
name = "${SOLUTION_NAME:-openshell-kda-solution-v1}"
definition = "${DEFINITION:-kernel_task}"
author = "${AUTHOR:-openshell}"

[build]
language = "${LANGUAGE:-triton}"
entry_point = "${normalized_entry_point}"
${source_dir_line}
${dps_line}
${binding_line}
EOF
}

sync_workspace_support_files() {
    local workspace_path="$1"
    local starter_path="$2"

    mkdir -p "${workspace_path}/scripts"

    for path in README.md FAQ.md EVALUATION.md; do
        if [[ -f "${starter_path}/${path}" ]]; then
            cp "${starter_path}/${path}" "${workspace_path}/${path}"
        fi
    done

    for path in check_cuda_extension.py pack_solution.py run_local.py run_modal.py; do
        if [[ -f "${starter_path}/scripts/${path}" ]]; then
            cp "${starter_path}/scripts/${path}" "${workspace_path}/scripts/${path}"
        fi
    done

    if [[ -d "${starter_path}/images" ]]; then
        mkdir -p "${workspace_path}/images"
        cp -R "${starter_path}/images/." "${workspace_path}/images/"
    fi
}

seed_definition_solution_if_available() {
    local workspace_path="$1"
    local example_root="$2"
    local definition_name="$3"
    local force_seed="${4:-0}"
    local template_root="${example_root}/definition-templates/${definition_name}"
    local starter_kernel="${example_root}/starter-kit/solution/cuda/kernel.cu"
    local workspace_kernel="${workspace_path}/solution/cuda/kernel.cu"

    if [[ -z "${definition_name}" || ! -d "${template_root}" ]]; then
        return 0
    fi

    if [[ "${force_seed}" != "1" ]]; then
        if [[ ! -f "${workspace_kernel}" ]]; then
            force_seed=1
        elif [[ -f "${starter_kernel}" ]] && cmp -s "${workspace_kernel}" "${starter_kernel}"; then
            force_seed=1
        elif grep -q "CUDA Kernel Template for FlashInfer Competition" "${workspace_kernel}" 2>/dev/null; then
            force_seed=1
        fi
    fi

    if [[ "${force_seed}" != "1" ]]; then
        return 0
    fi

    while IFS= read -r template_file; do
        local rel_path="${template_file#"${template_root}/"}"
        local target_path="${workspace_path}/${rel_path}"
        mkdir -p "$(dirname "${target_path}")"
        cp "${template_file}" "${target_path}"
    done < <(find "${template_root}" -type f | sort)
}

write_task_context_if_available() {
    local workspace_path="$1"
    local dataset_path="$2"
    local definition_name="$3"
    local workload_uuids="$4"
    local context_path="${workspace_path}/docs/task-context.md"

    if [[ -z "${dataset_path}" || -z "${definition_name}" || ! -d "${dataset_path}" ]]; then
        return 0
    fi

    python3 - "${dataset_path}" "${definition_name}" "${workload_uuids}" "${context_path}" <<'PY'
import json
import pathlib
import sys

dataset_root = pathlib.Path(sys.argv[1])
definition_name = sys.argv[2]
requested_uuids = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
output_path = pathlib.Path(sys.argv[4])


def find_definition(root: pathlib.Path, name: str):
    matches = list(root.glob(f"definitions/*/{name}.json"))
    return matches[0] if matches else None


def find_workload_file(root: pathlib.Path, name: str):
    matches = list(root.glob(f"workloads/*/{name}.jsonl"))
    return matches[0] if matches else None


def find_baseline_solution(root: pathlib.Path, name: str):
    matches = list(root.glob(f"solutions/baseline/*/{name}/*.json"))
    return matches[0] if matches else None


def shorten(text: str, limit: int = 1600) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


definition_path = find_definition(dataset_root, definition_name)
workload_path = find_workload_file(dataset_root, definition_name)
baseline_path = find_baseline_solution(dataset_root, definition_name)

if definition_path is None or workload_path is None:
    raise SystemExit(0)

definition = json.load(open(definition_path, encoding="utf-8"))
workloads = [json.loads(line) for line in open(workload_path, encoding="utf-8")]
selected = []
if requested_uuids:
    wanted = set(requested_uuids)
    for row in workloads:
        uuid = row.get("uuid") or row.get("workload", {}).get("uuid")
        if uuid in wanted:
            selected.append(row)
else:
    selected = workloads[:1]

lines = []
lines.append("# Task Context")
lines.append("")
lines.append("This file is generated from the local MLSys26 dataset for the active definition.")
lines.append("Use it as the first source for task semantics, selected workloads, and baseline context before exploring library internals.")
lines.append("")
lines.append("## Definition")
lines.append("")
lines.append(f"- Definition: `{definition.get('name', definition_name)}`")
lines.append(f"- Family: `{definition_path.parent.name}`")
lines.append(f"- op_type: `{definition.get('op_type', '<unknown>')}`")

axes = definition.get("axes", {})
if axes:
    lines.append("- Axes:")
    for key, spec in axes.items():
        axis_type = spec.get("type", "<unknown>")
        desc = spec.get("description", "")
        value = spec.get("value")
        suffix = f", value={value}" if value is not None else ""
        lines.append(f"  - `{key}`: type={axis_type}{suffix}; {desc}".rstrip())

inputs = definition.get("inputs", {})
if inputs:
    lines.append("- Inputs:")
    for key, spec in inputs.items():
        shape = spec.get("shape")
        dtype = spec.get("dtype", "<unknown>")
        desc = spec.get("description", "")
        lines.append(f"  - `{key}`: dtype={dtype}, shape={shape}; {desc}".rstrip())

outputs = definition.get("outputs", {})
if outputs:
    lines.append("- Outputs:")
    for key, spec in outputs.items():
        shape = spec.get("shape")
        dtype = spec.get("dtype", "<unknown>")
        desc = spec.get("description", "")
        lines.append(f"  - `{key}`: dtype={dtype}, shape={shape}; {desc}".rstrip())

reference = definition.get("reference")
if isinstance(reference, str) and reference.strip():
    lines.append("")
    lines.append("## Reference Summary")
    lines.append("")
    lines.append("```python")
    lines.append(shorten(reference, 2200))
    lines.append("```")

lines.append("")
lines.append("## Selected Workloads")
lines.append("")
if selected:
    for row in selected:
        workload = row.get("workload", row)
        uuid = workload.get("uuid", "<unknown>")
        axes_map = workload.get("axes", {})
        lines.append(f"- UUID: `{uuid}`")
        if axes_map:
            lines.append(f"  - Axes: `{json.dumps(axes_map, sort_keys=True)}`")
        inputs_map = workload.get("inputs", {})
        if inputs_map:
            summarized = {}
            for key, value in inputs_map.items():
                if isinstance(value, dict):
                    summarized[key] = {k: value[k] for k in value if k in {'type', 'path', 'tensor_key', 'value'}}
                else:
                    summarized[key] = value
            lines.append(f"  - Inputs: `{json.dumps(summarized, sort_keys=True)}`")
else:
    lines.append("- No matching workloads were found for the configured UUID subset.")

if baseline_path is not None:
    baseline = json.load(open(baseline_path, encoding="utf-8"))
    lines.append("")
    lines.append("## Baseline Solution")
    lines.append("")
    lines.append(f"- File: `{baseline_path.relative_to(dataset_root)}`")
    lines.append(f"- Name: `{baseline.get('name', '<unknown>')}`")
    lines.append(f"- Author: `{baseline.get('author', '<unknown>')}`")
    description = baseline.get("description")
    if description:
        lines.append(f"- Description: {description}")
    sources = baseline.get("sources", [])
    if sources:
        lines.append("- Sources:")
        for source in sources:
            path = source.get("path") or source.get("filename") or source.get("name") or "<unknown>"
            lines.append(f"  - `{path}`")
        primary = sources[0].get("content")
        if isinstance(primary, str) and primary.strip():
            lines.append("")
            lines.append("```python")
            lines.append(shorten(primary, 2200))
            lines.append("```")

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

rewrite_dataset_placeholders() {
    local command_text="$1"
    local dataset_path="$2"

    command_text="${command_text//\/path\/to\/mlsys26-contest/${dataset_path}}"
    command_text="${command_text//\/path\/to\/flashinfer-trace/${dataset_path}}"
    printf '%s\n' "${command_text}"
}

normalize_validation_command() {
    local command_text="$1"

    if [[ -z "${command_text}" ]]; then
        printf '%s\n' "${command_text}"
        return 0
    fi

    if [[ "${command_text}" == *"check_cuda_extension.py"* && "${command_text}" != *"--log-file"* ]]; then
        command_text="${command_text} --log-file runs/check_cuda_extension.txt"
    fi

    printf '%s\n' "${command_text}"
}

normalize_evaluation_command() {
    local command_text="$1"

    if [[ -z "${command_text}" ]]; then
        printf '%s\n' "${command_text}"
        return 0
    fi

    if [[ "${command_text}" == *"run_local.py"* ]]; then
        if [[ "${command_text}" != *"--log-file"* ]]; then
            command_text="${command_text} --log-file runs/run_local.txt"
        fi
        if [[ "${command_text}" != *"--results-json"* ]]; then
            command_text="${command_text} --results-json runs/run_local_results.json"
        fi
    fi

    printf '%s\n' "${command_text}"
}

for arg in "$@"; do
    case "${arg}" in
        --config=*)
            CONFIG_PATH="${arg#--config=}"
            ;;
    esac
done

CONFIG_PATH="$(to_abs_path "${CONFIG_PATH}")"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 2
fi

load_simple_yaml_config "${CONFIG_PATH}"

for arg in "$@"; do
    case "${arg}" in
        --config=*)
            ;;
        --workspace=*)
            WORKSPACE="${arg#--workspace=}"
            ;;
        --starter-kit=*)
            STARTER_KIT_DIR="${arg#--starter-kit=}"
            ;;
        --gpus=*)
            GPU_SPEC="${arg#--gpus=}"
            ;;
        --model=*)
            MODEL="${arg#--model=}"
            ;;
        --reasoning=*)
            REASONING="${arg#--reasoning=}"
            ;;
        --image=*)
            IMAGE="${arg#--image=}"
            ;;
        --mode=*)
            MODE="${arg#--mode=}"
            ;;
        --execute)
            MODE="execute"
            ;;
        --docker-user=*)
            DOCKER_USER="${arg#--docker-user=}"
            ;;
        --host-runtime-root=*)
            HOST_RUNTIME_ROOT="${arg#--host-runtime-root=}"
            ;;
        --fib-dataset-path=*)
            FIB_DATASET_PATH_VALUE="${arg#--fib-dataset-path=}"
            ;;
        --build)
            BUILD_IMAGE=1
            ;;
        --kernel-optimize)
            KERNEL_OPTIMIZE=1
            ;;
        --max-depth=*)
            MAX_DEPTH="${arg#--max-depth=}"
            ;;
        --branch=*)
            BRANCH="${arg#--branch=}"
            ;;
        --no-scaffold)
            AUTO_SCAFFOLD=0
            ;;
        --clean-workspace|--force-scaffold)
            FORCE_SCAFFOLD=1
            ;;
        --solution-name=*)
            SOLUTION_NAME="${arg#--solution-name=}"
            ;;
        --definition=*)
            DEFINITION="${arg#--definition=}"
            ;;
        --author=*)
            AUTHOR="${arg#--author=}"
            ;;
        --language=*)
            LANGUAGE="${arg#--language=}"
            ;;
        --entry-point=*)
            ENTRY_POINT="${arg#--entry-point=}"
            ;;
        --source-dir=*)
            SOURCE_DIR="${arg#--source-dir=}"
            ;;
        --destination-passing-style=*)
            DESTINATION_PASSING_STYLE="${arg#--destination-passing-style=}"
            ;;
        --binding=*)
            BINDING="${arg#--binding=}"
            ;;
        --benchmark-warmup-runs=*)
            BENCHMARK_WARMUP_RUNS="${arg#--benchmark-warmup-runs=}"
            ;;
        --benchmark-iterations=*)
            BENCHMARK_ITERATIONS="${arg#--benchmark-iterations=}"
            ;;
        --benchmark-num-trials=*)
            BENCHMARK_NUM_TRIALS="${arg#--benchmark-num-trials=}"
            ;;
        --benchmark-workload-limit=*)
            BENCHMARK_WORKLOAD_LIMIT="${arg#--benchmark-workload-limit=}"
            ;;
        --benchmark-workload-uuids=*)
            BENCHMARK_WORKLOAD_UUIDS="${arg#--benchmark-workload-uuids=}"
            ;;
        --task-name=*)
            TASK_NAME="${arg#--task-name=}"
            ;;
        --objective=*)
            OBJECTIVE="${arg#--objective=}"
            ;;
        --correctness=*)
            CORRECTNESS="${arg#--correctness=}"
            ;;
        --target=*)
            TARGET="${arg#--target=}"
            ;;
        --allowed=*)
            ALLOWED="${arg#--allowed=}"
            ;;
        --validate=*)
            VALIDATE="${arg#--validate=}"
            ;;
        --evaluate=*)
            EVALUATE="${arg#--evaluate=}"
            ;;
        --promote=*)
            PROMOTE="${arg#--promote=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

WORKSPACE="$(to_abs_path "${WORKSPACE}")"
STARTER_KIT_DIR="$(to_abs_path "${STARTER_KIT_DIR}")"
HOST_RUNTIME_ROOT="$(to_abs_path "${HOST_RUNTIME_ROOT}")"
MODE="$(normalize_mode "${MODE}")"

case "${MAX_DEPTH}" in
    ''|*[!0-9]*)
        echo "Unsupported max depth: ${MAX_DEPTH}. Expected a positive integer." >&2
        exit 2
        ;;
    0)
        echo "Unsupported max depth: 0. Expected a positive integer." >&2
        exit 2
        ;;
esac

case "${BRANCH}" in
    ''|*[!0-9]*)
        echo "Unsupported branch count: ${BRANCH}. Expected a positive integer." >&2
        exit 2
        ;;
    0)
        echo "Unsupported branch count: 0. Expected a positive integer." >&2
        exit 2
        ;;
esac

if [[ -n "${FIB_DATASET_PATH_VALUE}" ]]; then
    FIB_DATASET_PATH_VALUE="$(to_abs_path "${FIB_DATASET_PATH_VALUE}")"
fi

if [[ ! -d "${STARTER_KIT_DIR}" ]]; then
    echo "Starter-kit directory not found: ${STARTER_KIT_DIR}" >&2
    exit 2
fi

if [[ -n "${FIB_DATASET_PATH_VALUE}" && ! -d "${FIB_DATASET_PATH_VALUE}" ]]; then
    if [[ "${MODE}" == "execute" ]]; then
        echo "FlashInfer dataset directory not found: ${FIB_DATASET_PATH_VALUE}" >&2
        exit 2
    fi
    echo "Warning: dataset directory not found for draft mode; continuing without dataset mount: ${FIB_DATASET_PATH_VALUE}" >&2
    FIB_DATASET_PATH_VALUE=""
fi

if [[ "${AUTO_SCAFFOLD}" == "1" ]]; then
    if [[ ! -f "${WORKSPACE}/TASK_CONTRACT.md" || ! -f "${WORKSPACE}/config.toml" || "${FORCE_SCAFFOLD}" == "1" ]]; then
        scaffold_cmd=(bash "${EXAMPLE_DIR}/scaffold.sh")
        if [[ "${FORCE_SCAFFOLD}" == "1" ]]; then
            scaffold_cmd+=(--force)
        fi
        scaffold_cmd+=(--template="${STARTER_KIT_DIR}" "${WORKSPACE}")
        mkdir -p "$(dirname "${WORKSPACE}")"
        "${scaffold_cmd[@]}"
        SCAFFOLDED_WORKSPACE=1
    fi
fi

sync_workspace_support_files "${WORKSPACE}" "${STARTER_KIT_DIR}"
ACTIVE_DEFINITION_FOR_SETUP="${DEFINITION}"
if [[ -z "${ACTIVE_DEFINITION_FOR_SETUP}" ]]; then
    ACTIVE_DEFINITION_FOR_SETUP="$(read_workspace_definition "${WORKSPACE}/config.toml" || true)"
fi
seed_definition_solution_if_available "${WORKSPACE}" "${EXAMPLE_DIR}" "${ACTIVE_DEFINITION_FOR_SETUP}" "$(( FORCE_SCAFFOLD == 1 || SCAFFOLDED_WORKSPACE == 1 ))"

if [[ ! -f "${WORKSPACE}/TASK_CONTRACT.md" ]]; then
    echo "Missing TASK_CONTRACT.md: ${WORKSPACE}" >&2
    echo "Either create the workspace first or omit --no-scaffold." >&2
    exit 2
fi

if [[ ! -f "${WORKSPACE}/config.toml" ]]; then
    echo "Missing starter-kit config.toml: ${WORKSPACE}" >&2
    echo "Either create the workspace from the starter kit or omit --no-scaffold." >&2
    exit 2
fi

if [[ -n "${FIB_DATASET_PATH_VALUE}" && -n "${EVALUATE}" ]]; then
    EVALUATE="$(rewrite_dataset_placeholders "${EVALUATE}" "${CONTAINER_FIB_DATASET_PATH}")"
fi
VALIDATE="$(normalize_validation_command "${VALIDATE}")"
EVALUATE="$(normalize_evaluation_command "${EVALUATE}")"

write_contract_if_requested "${WORKSPACE}/TASK_CONTRACT.md"
write_starter_config_if_requested "${WORKSPACE}/config.toml"
ACTIVE_DEFINITION_FOR_SETUP="${DEFINITION}"
if [[ -z "${ACTIVE_DEFINITION_FOR_SETUP}" ]]; then
    ACTIVE_DEFINITION_FOR_SETUP="$(read_workspace_definition "${WORKSPACE}/config.toml" || true)"
fi
write_task_context_if_available "${WORKSPACE}" "${FIB_DATASET_PATH_VALUE}" "${ACTIVE_DEFINITION_FOR_SETUP}" "${BENCHMARK_WORKLOAD_UUIDS}"

if [[ -n "${OPENAI_API_KEY_VALUE}" && -n "${CODEX_API_KEY_VALUE}" ]]; then
    echo "Set only one of OPENAI_API_KEY or CODEX_API_KEY before running optimize.sh." >&2
    exit 2
elif [[ -n "${OPENAI_API_KEY_VALUE}" || -n "${CODEX_API_KEY_VALUE}" ]]; then
    AUTH_MODE="api-key"
    if [[ -z "${OPENAI_API_KEY_VALUE}" ]]; then
        OPENAI_API_KEY_VALUE="${CODEX_API_KEY_VALUE}"
    fi
    if [[ -z "${CODEX_API_KEY_VALUE}" ]]; then
        CODEX_API_KEY_VALUE="${OPENAI_API_KEY_VALUE}"
    fi
else
    AUTH_MODE="oauth"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for examples/kernel-design-agents/optimize.sh" >&2
    exit 2
fi

if [[ "${AUTH_MODE}" == "oauth" ]]; then
    if ! command -v jq >/dev/null 2>&1; then
        echo "jq is required when using ~/.codex/auth.json" >&2
        exit 2
    fi
    if [[ ! -f "${HOME}/.codex/auth.json" ]]; then
        echo "Set OPENAI_API_KEY/CODEX_API_KEY or run codex login first." >&2
        exit 2
    fi
    CODEX_AUTH_ACCESS_TOKEN="$(jq -r '.tokens.access_token // empty' "${HOME}/.codex/auth.json")"
    CODEX_AUTH_REFRESH_TOKEN="$(jq -r '.tokens.refresh_token // empty' "${HOME}/.codex/auth.json")"
    CODEX_AUTH_ACCOUNT_ID="$(jq -r '.tokens.account_id // empty' "${HOME}/.codex/auth.json")"
    if [[ -z "${CODEX_AUTH_ACCESS_TOKEN}" || -z "${CODEX_AUTH_REFRESH_TOKEN}" || -z "${CODEX_AUTH_ACCOUNT_ID}" ]]; then
        echo "Local Codex auth.json is missing one or more required fields." >&2
        exit 2
    fi
fi

if [[ "${BUILD_IMAGE}" == "1" ]]; then
    echo "Building image: ${IMAGE}"
    docker build -t "${IMAGE}" "${EXAMPLE_DIR}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Docker image not found: ${IMAGE}"
    echo "Building it now..."
    docker build -t "${IMAGE}" "${EXAMPLE_DIR}"
fi

ABS_WORKSPACE="$(cd "${WORKSPACE}" && pwd)"
mkdir -p "${HOST_RUNTIME_ROOT}"
RUN_DIR="$(mktemp -d "${HOST_RUNTIME_ROOT%/}/run.XXXXXX")"
CONTAINER_WORKSPACE="/workspace/task"
CONTAINER_RUNTIME_ROOT="/runtime/kda"
DOCKER_GPUS="$(normalize_docker_gpus "${GPU_SPEC}")"

cleanup() {
    rm -rf "${RUN_DIR}"
}
trap cleanup EXIT

docker_mounts=(
    -v "${ABS_WORKSPACE}:${CONTAINER_WORKSPACE}"
    -v "${RUN_DIR}:${CONTAINER_RUNTIME_ROOT}"
)

if [[ -n "${FIB_DATASET_PATH_VALUE}" ]]; then
    docker_mounts+=(-v "${FIB_DATASET_PATH_VALUE}:${CONTAINER_FIB_DATASET_PATH}:ro")
fi

docker_env=(
    -e HOME="${CONTAINER_RUNTIME_ROOT}/home"
    -e TMPDIR="${CONTAINER_RUNTIME_ROOT}/tmp"
    -e TMP="${CONTAINER_RUNTIME_ROOT}/tmp"
    -e TEMP="${CONTAINER_RUNTIME_ROOT}/tmp"
    -e XDG_CONFIG_HOME="${CONTAINER_RUNTIME_ROOT}/config"
    -e XDG_CACHE_HOME="${CONTAINER_RUNTIME_ROOT}/cache"
    -e XDG_DATA_HOME="${CONTAINER_RUNTIME_ROOT}/data"
    -e XDG_STATE_HOME="${CONTAINER_RUNTIME_ROOT}/state"
    -e CODEX_HOME="${CONTAINER_RUNTIME_ROOT}/codex"
    -e CODEX_SQLITE_HOME="${CONTAINER_RUNTIME_ROOT}/codex/sqlite"
    -e KDA_CODEX_RUNTIME_ROOT="${CONTAINER_RUNTIME_ROOT}"
    -e KDA_EXECUTION_MODE="${MODE}"
    -e KDA_KERNEL_OPTIMIZE="${KERNEL_OPTIMIZE}"
    -e KDA_MAX_DEPTH="${MAX_DEPTH}"
    -e KDA_BRANCH="${BRANCH}"
)

if [[ -n "${FIB_DATASET_PATH_VALUE}" ]]; then
    docker_env+=(-e FIB_DATASET_PATH="${CONTAINER_FIB_DATASET_PATH}")
fi

if [[ -n "${BENCHMARK_WARMUP_RUNS}" ]]; then
    docker_env+=(-e KDA_BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS}")
fi
if [[ -n "${BENCHMARK_ITERATIONS}" ]]; then
    docker_env+=(-e KDA_BENCHMARK_ITERATIONS="${BENCHMARK_ITERATIONS}")
fi
if [[ -n "${BENCHMARK_NUM_TRIALS}" ]]; then
    docker_env+=(-e KDA_BENCHMARK_NUM_TRIALS="${BENCHMARK_NUM_TRIALS}")
fi
if [[ -n "${BENCHMARK_WORKLOAD_LIMIT}" ]]; then
    docker_env+=(-e KDA_BENCHMARK_WORKLOAD_LIMIT="${BENCHMARK_WORKLOAD_LIMIT}")
fi
if [[ -n "${BENCHMARK_WORKLOAD_UUIDS}" ]]; then
    docker_env+=(-e KDA_BENCHMARK_WORKLOAD_UUIDS="${BENCHMARK_WORKLOAD_UUIDS}")
fi

if [[ "${AUTH_MODE}" == "api-key" ]]; then
    docker_env+=(
        -e OPENAI_API_KEY="${OPENAI_API_KEY_VALUE}"
        -e CODEX_API_KEY="${CODEX_API_KEY_VALUE}"
        -e CODEX_ACCESS_TOKEN=
    )
else
    docker_env+=(
        -e CODEX_AUTH_ACCESS_TOKEN="${CODEX_AUTH_ACCESS_TOKEN}"
        -e CODEX_AUTH_REFRESH_TOKEN="${CODEX_AUTH_REFRESH_TOKEN}"
        -e CODEX_AUTH_ACCOUNT_ID="${CODEX_AUTH_ACCOUNT_ID}"
    )
fi

echo "Running KDA optimize entrypoint"
echo "  Config: ${CONFIG_PATH}"
if [[ -n "${PRESET_NAME}" ]]; then
    echo "  Preset: ${PRESET_NAME}"
fi
if [[ -n "${TASK_FAMILY}" ]]; then
    echo "  Task family: ${TASK_FAMILY}"
fi
if [[ -n "${DEFINITION}" ]]; then
    echo "  Definition: ${DEFINITION}"
fi
echo "  Starter kit: ${STARTER_KIT_DIR}"
echo "  Workspace: ${ABS_WORKSPACE}"
echo "  Model: ${MODEL}"
echo "  Reasoning: ${REASONING}"
echo "  Mode: ${MODE}"
echo "  Kernel optimize: ${KERNEL_OPTIMIZE}"
echo "  Max depth: ${MAX_DEPTH}"
echo "  Branch: ${BRANCH}"
echo "  Image: ${IMAGE}"
echo "  Docker user: ${DOCKER_USER}"
echo "  Docker GPUs: ${DOCKER_GPUS}"
echo "  Auth mode: ${AUTH_MODE}"
if [[ -n "${LANGUAGE}${ENTRY_POINT}${SOURCE_DIR}${BINDING}" ]]; then
    echo "  Build: language=${LANGUAGE:-default} entry_point=${ENTRY_POINT:-default} source_dir=${SOURCE_DIR:-default} binding=${BINDING:-default}"
fi
if [[ -n "${FIB_DATASET_PATH_VALUE}" ]]; then
    echo "  Dataset: ${FIB_DATASET_PATH_VALUE} -> ${CONTAINER_FIB_DATASET_PATH}"
elif [[ "${MODE}" == "execute" ]]; then
    echo "  Dataset: not set (evaluation may be skipped or reported as blocked)"
fi
if [[ -n "${WORKLOAD_PROFILE}" ]]; then
    echo "  Workload profile: ${WORKLOAD_PROFILE}"
fi
if [[ -n "${BENCHMARK_WORKLOAD_UUIDS}" ]]; then
    echo "  Benchmark workloads: ${BENCHMARK_WORKLOAD_UUIDS}"
elif [[ -n "${BENCHMARK_WORKLOAD_LIMIT}" ]]; then
    echo "  Benchmark workload limit: ${BENCHMARK_WORKLOAD_LIMIT}"
elif [[ "${WORKLOAD_PROFILE}" == "wa" ]]; then
    echo "  Benchmark workloads: all"
fi
if [[ -n "${BENCHMARK_WARMUP_RUNS}${BENCHMARK_ITERATIONS}${BENCHMARK_NUM_TRIALS}" ]]; then
    echo "  Benchmark config: warmup=${BENCHMARK_WARMUP_RUNS:-default} iterations=${BENCHMARK_ITERATIONS:-default} trials=${BENCHMARK_NUM_TRIALS:-default}"
fi

docker run --rm \
    --gpus "${DOCKER_GPUS}" \
    --user "${DOCKER_USER}" \
    --workdir "${CONTAINER_WORKSPACE}" \
    "${docker_mounts[@]}" \
    "${docker_env[@]}" \
    "${IMAGE}" \
    /bin/bash /app/scripts/run-kda-draft.sh \
    --workspace "${CONTAINER_WORKSPACE}" \
    --model "${MODEL}" \
    --reasoning "${REASONING}" \
    --mode "${MODE}" \
    --runtime-root "${CONTAINER_RUNTIME_ROOT}"

echo
echo "KDA ${MODE} run completed."
echo "Workspace: ${ABS_WORKSPACE}"
if [[ "${MODE}" == "execute" ]]; then
    echo "Attempt workspaces: ${ABS_WORKSPACE}/baseline/b* and ${ABS_WORKSPACE}/d*/b*"
    echo "Current pointer: ${ABS_WORKSPACE}/current.json"
    echo "Agent summary: ${ABS_WORKSPACE}/outputs/execution-summary.agent.md"
    echo "Execution summary: ${ABS_WORKSPACE}/outputs/execution-summary.md"
else
    echo "Draft: ${ABS_WORKSPACE}/docs/draft.md"
fi
echo "Last message: ${ABS_WORKSPACE}/outputs/last-message.md"
