#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="${KDA_EXAMPLE_DIR:-${ROOT}/examples/kernel-design-agents}"
CONFIG_PATH="${KDA_CONFIG:-${EXAMPLE_DIR}/config/kda-gemm-task.yml}"
WORKSPACE="${KDA_WORKSPACE:-${EXAMPLE_DIR}/results/kda-flashinfer-task}"
STARTER_KIT_DIR="${KDA_STARTER_KIT_DIR:-${EXAMPLE_DIR}/starter-kit}"
GPU_SPEC="${KDA_GPU_SPEC:-1}"
MODEL="${KDA_CODEX_MODEL:-gpt-5.4-mini}"
REASONING="${KDA_CODEX_REASONING:-low}"
IMAGE="${KDA_IMAGE:-openshell-kda-example}"
DOCKER_USER_DEFAULT="$(id -u):$(id -g)"
DOCKER_USER="${KDA_DOCKER_USER:-${DOCKER_USER_DEFAULT}}"
HOST_RUNTIME_ROOT="${KDA_HOST_RUNTIME_ROOT:-${TMPDIR:-/tmp}/openshell-kda-runtime}"
AUTO_SCAFFOLD=1
FORCE_SCAFFOLD=0
BUILD_IMAGE=0

SOLUTION_NAME="${KDA_SOLUTION_NAME:-}"
DEFINITION="${KDA_DEFINITION:-}"
AUTHOR="${KDA_AUTHOR:-}"
LANGUAGE="${KDA_LANGUAGE:-}"
ENTRY_POINT="${KDA_ENTRY_POINT:-}"

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
into the example image and runs the draft loop directly against host files.

Options:
  --config=<path>         YAML config file. Default: examples/kernel-design-agents/config/kda-gemm-task.yml
  --workspace=<path>      Local task workspace.
  --starter-kit=<path>    Starter-kit template source. Default: examples/kernel-design-agents/starter-kit
  --gpus=<spec>           Docker GPU selector. Examples: 1, all, device=1
  --model=<name>          Codex model override.
  --reasoning=<level>     Codex reasoning effort. Default: low
  --image=<tag>           Local Docker image tag. Default: openshell-kda-example
  --docker-user=<u:g>     Container user. Default: current host uid:gid
  --host-runtime-root=<p> Host directory for temporary Codex runtime state
  --build                 Build the example image before running.
  --no-scaffold           Require an existing workspace and TASK_CONTRACT.md.
  --force-scaffold        Recreate scaffolded files before the run.
  --solution-name=<text>  Write workspace config.toml automatically.
  --definition=<text>     Write workspace config.toml automatically.
  --author=<text>         Write workspace config.toml automatically.
  --language=<text>       Write workspace config.toml automatically.
  --entry-point=<text>    Write workspace config.toml automatically.
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
            docker_user) DOCKER_USER="${value}" ;;
            host_runtime_root) HOST_RUNTIME_ROOT="${value}" ;;
            auto_scaffold) [[ "${value}" == "false" ]] && AUTO_SCAFFOLD=0 || AUTO_SCAFFOLD=1 ;;
            force_scaffold) [[ "${value}" == "true" ]] && FORCE_SCAFFOLD=1 || FORCE_SCAFFOLD=0 ;;
            build_image) [[ "${value}" == "true" ]] && BUILD_IMAGE=1 || BUILD_IMAGE=0 ;;
            solution_name) SOLUTION_NAME="${value}" ;;
            definition) DEFINITION="${value}" ;;
            author) AUTHOR="${value}" ;;
            language) LANGUAGE="${value}" ;;
            entry_point) ENTRY_POINT="${value}" ;;
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

    if [[ -z "${SOLUTION_NAME}" && -z "${DEFINITION}" && -z "${AUTHOR}" && -z "${LANGUAGE}" && -z "${ENTRY_POINT}" ]]; then
        return 0
    fi

    cat > "${config_path}" <<EOF
[solution]
name = "${SOLUTION_NAME:-my-team-solution-v1}"
definition = "${DEFINITION:-fused_moe}"
author = "${AUTHOR:-team-name}"

[build]
language = "${LANGUAGE:-triton}"
entry_point = "${ENTRY_POINT:-kernel}"
EOF
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
        --docker-user=*)
            DOCKER_USER="${arg#--docker-user=}"
            ;;
        --host-runtime-root=*)
            HOST_RUNTIME_ROOT="${arg#--host-runtime-root=}"
            ;;
        --build)
            BUILD_IMAGE=1
            ;;
        --no-scaffold)
            AUTO_SCAFFOLD=0
            ;;
        --force-scaffold)
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

if [[ ! -d "${STARTER_KIT_DIR}" ]]; then
    echo "Starter-kit directory not found: ${STARTER_KIT_DIR}" >&2
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
)

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
echo "  Image: ${IMAGE}"
echo "  Docker user: ${DOCKER_USER}"
echo "  Docker GPUs: ${DOCKER_GPUS}"
echo "  Auth mode: ${AUTH_MODE}"

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
    --runtime-root "${CONTAINER_RUNTIME_ROOT}"

echo
echo "KDA draft completed."
echo "Workspace: ${ABS_WORKSPACE}"
echo "Draft: ${ABS_WORKSPACE}/docs/draft.md"
echo "Last message: ${ABS_WORKSPACE}/outputs/last-message.md"
