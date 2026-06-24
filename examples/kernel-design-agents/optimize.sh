#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="${KDA_EXAMPLE_DIR:-${ROOT}/examples/kernel-design-agents}"
CONFIG_PATH="${KDA_CONFIG:-${EXAMPLE_DIR}/config/kda-gemm-task.yml}"
WORKSPACE="${KDA_WORKSPACE:-${EXAMPLE_DIR}/results/kda-flashinfer-moe-phase1}"
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
BUILD_IMAGE=0
KERNEL_OPTIMIZE=0
MAX_DEPTH="${KDA_MAX_DEPTH:-1}"

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
into the example image and runs either a draft-only or implementation loop
directly against host files.

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
  --kernel-optimize       If a validated baseline exists, optimize it for
                          latency. Otherwise bootstrap a baseline first, then
                          attempt one optimization candidate in the same run.
  --max-depth=<n>         Maximum number of optimization candidates to attempt
                          in this run. Default: 1.
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
name = "${SOLUTION_NAME:-my-team-solution-v1}"
definition = "${DEFINITION:-fused_moe}"
author = "${AUTHOR:-team-name}"

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

rewrite_dataset_placeholders() {
    local command_text="$1"
    local dataset_path="$2"

    command_text="${command_text//\/path\/to\/mlsys26-contest/${dataset_path}}"
    command_text="${command_text//\/path\/to\/flashinfer-trace/${dataset_path}}"
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

if [[ -n "${FIB_DATASET_PATH_VALUE}" ]]; then
    FIB_DATASET_PATH_VALUE="$(to_abs_path "${FIB_DATASET_PATH_VALUE}")"
fi

if [[ ! -d "${STARTER_KIT_DIR}" ]]; then
    echo "Starter-kit directory not found: ${STARTER_KIT_DIR}" >&2
    exit 2
fi

if [[ -n "${FIB_DATASET_PATH_VALUE}" && ! -d "${FIB_DATASET_PATH_VALUE}" ]]; then
    echo "FlashInfer dataset directory not found: ${FIB_DATASET_PATH_VALUE}" >&2
    exit 2
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
    fi
fi

sync_workspace_support_files "${WORKSPACE}" "${STARTER_KIT_DIR}"

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

write_contract_if_requested "${WORKSPACE}/TASK_CONTRACT.md"
write_starter_config_if_requested "${WORKSPACE}/config.toml"

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
    -e KDA_RUN_LOCAL_LOG_PATH="${CONTAINER_WORKSPACE}/runs/run_local.txt"
    -e KDA_RUN_LOCAL_RESULTS_PATH="${CONTAINER_WORKSPACE}/runs/run_local_results.json"
    -e KDA_CUDA_EXTENSION_LOG_PATH="${CONTAINER_WORKSPACE}/runs/check_cuda_extension.txt"
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
echo "  Starter kit: ${STARTER_KIT_DIR}"
echo "  Workspace: ${ABS_WORKSPACE}"
echo "  Model: ${MODEL}"
echo "  Reasoning: ${REASONING}"
echo "  Mode: ${MODE}"
echo "  Kernel optimize: ${KERNEL_OPTIMIZE}"
echo "  Max depth: ${MAX_DEPTH}"
echo "  Image: ${IMAGE}"
echo "  Docker user: ${DOCKER_USER}"
echo "  Docker GPUs: ${DOCKER_GPUS}"
echo "  Auth mode: ${AUTH_MODE}"
if [[ -n "${FIB_DATASET_PATH_VALUE}" ]]; then
    echo "  Dataset: ${FIB_DATASET_PATH_VALUE} -> ${CONTAINER_FIB_DATASET_PATH}"
elif [[ "${MODE}" == "execute" ]]; then
    echo "  Dataset: not set (evaluation may be skipped or reported as blocked)"
fi
if [[ -n "${BENCHMARK_WORKLOAD_UUIDS}" ]]; then
    echo "  Benchmark workloads: ${BENCHMARK_WORKLOAD_UUIDS}"
elif [[ -n "${BENCHMARK_WORKLOAD_LIMIT}" ]]; then
    echo "  Benchmark workload limit: ${BENCHMARK_WORKLOAD_LIMIT}"
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
echo "Draft: ${ABS_WORKSPACE}/docs/draft.md"
if [[ "${MODE}" == "execute" ]]; then
    echo "Plan: ${ABS_WORKSPACE}/docs/plan.md"
    echo "Execution summary: ${ABS_WORKSPACE}/outputs/execution-summary.md"
fi
echo "Last message: ${ABS_WORKSPACE}/outputs/last-message.md"
