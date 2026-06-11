#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="${ROOT}/examples/autoagent"

IMAGE="${AUTOAGENT_IMAGE:-openshell-autoagent}"
CONFIG_PATH="${AUTOAGENT_CONFIG:-configs/softmax.yaml}"
HOST_STATE_DIR="${AUTOAGENT_HOST_STATE_DIR:-${EXAMPLE_DIR}/results}"
CONTAINER_STATE_DIR="${AUTOAGENT_STATE_DIR:-/sandbox/autoagent}"
CONTAINER_WORKSPACE_DIR="${CONTAINER_STATE_DIR}/workspace"
HOST_RUNTIME_ROOT="${HOST_STATE_DIR}/tmp"
HOST_CONTAINER_HOME="${HOST_RUNTIME_ROOT}/home"
HOST_CONTAINER_TMP="${HOST_RUNTIME_ROOT}/tmp"
HOST_CONTAINER_XDG_CONFIG="${HOST_RUNTIME_ROOT}/config"
HOST_CONTAINER_XDG_CACHE="${HOST_RUNTIME_ROOT}/cache"
HOST_CONTAINER_XDG_DATA="${HOST_RUNTIME_ROOT}/data"
HOST_CONTAINER_XDG_STATE="${HOST_RUNTIME_ROOT}/state"
HOST_CODEX_STAGE_ROOT="${AUTOAGENT_HOST_CODEX_STAGE_ROOT:-${TMPDIR:-/tmp}/openshell-autoagent-codex}"
HOST_CODEX_STAGE_DIR="${HOST_CODEX_STAGE_ROOT}/run-$$"
HOST_CONTAINER_CODEX_HOME="${HOST_CODEX_STAGE_DIR}/codex"
CONTAINER_RUNTIME_ROOT="${CONTAINER_STATE_DIR}/tmp"
CONTAINER_HOME="${CONTAINER_RUNTIME_ROOT}/home"
CONTAINER_TMPDIR="${CONTAINER_RUNTIME_ROOT}/tmp"
CONTAINER_XDG_CONFIG="${CONTAINER_RUNTIME_ROOT}/config"
CONTAINER_XDG_CACHE="${CONTAINER_RUNTIME_ROOT}/cache"
CONTAINER_XDG_DATA="${CONTAINER_RUNTIME_ROOT}/data"
CONTAINER_XDG_STATE="${CONTAINER_RUNTIME_ROOT}/state"
CONTAINER_CODEX_HOME="${CONTAINER_RUNTIME_ROOT}/codex"
GPU_SPEC="${AUTOAGENT_GPU_SPEC:-1}"
BRANCH_FACTOR="${AUTOAGENT_BRANCH_FACTOR:-1}"
MAX_DEPTH="${AUTOAGENT_MAX_DEPTH:-0}"
MAX_RETRIES="${AUTOAGENT_MAX_RETRIES:-2}"
PROFILE_EXECUTOR="${AUTOAGENT_PROFILE_EXECUTOR:-ncu}"
NCU_COMMAND_IN_CONTAINER="${AUTOAGENT_NCU_COMMAND:-ncu}"
CODEX_COMMAND_IN_CONTAINER="${AUTOAGENT_CODEX_COMMAND:-}"
DOCKER_USER="${AUTOAGENT_DOCKER_USER:-0:0}"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY:-}"
CODEX_API_KEY_VALUE="${CODEX_API_KEY:-}"
CODEX_LOGIN="${AUTOAGENT_CODEX_LOGIN:-0}"
BUILD_IMAGE=0
CHECK_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash examples/autoagent/optimize.sh [options]

This runs AutoAgent in a dedicated profiling-capable Docker container rather
than through the standard OpenShell sandbox seccomp profile.

Options:
  --build                 Build the example image before running.
  --check                 Only run host/container preflight checks.
  --gpus=<spec>           Docker GPU selector. `--gpus=1` maps to `device=1`.
  --config=<path>         Kernel config inside /app. Default: configs/softmax.yaml
  --image=<tag>           Local Docker image tag. Default: openshell-autoagent
  --host-state-dir=<dir>  Host directory for kernels/, output/, and results.csv
  --state-dir=<path>      Writable state dir inside the container.
  --branch-factor=<n>     Candidate count per depth. Default: 1
  --max-depth=<n>         Search depth. Use 0 to only profile the baseline and
                          still emit kernels/results.csv. Default: 0
  --max-retries=<n>       Retry count per branch. Default: 2
  --docker-user=<u:g>     Container user for profiling runs. Default: 0:0
  --codex-login           Prompt for a temporary in-container `codex login
                          --with-api-key`. Credentials stay inside the
                          container runtime and are discarded after the run.
  --codex-command=<cmd>   Container command for Codex CLI. If omitted, the
                          script tries to mount a host `codex` binary.
  --ncu-command=<cmd>     Container command for Nsight Compute. Default: ncu
  -h, --help              Show this help.
EOF
}

normalize_docker_gpus() {
    local value="$1"
    if [[ "${value}" == "all" || "${value}" == *"="* ]]; then
        printf '%s\n' "${value}"
        return 0
    fi
    printf 'device=%s\n' "${value}"
}

resolve_real_path() {
    local path="$1"
    if command -v readlink >/dev/null 2>&1; then
        readlink -f "${path}" 2>/dev/null || printf '%s\n' "${path}"
        return 0
    fi
    printf '%s\n' "${path}"
}

mount_host_command_if_needed() {
    local raw_command="$1"
    local mount_name="$2"
    local -a command_parts=()
    local executable=""
    local tail_args=""
    local resolved=""
    local resolved_real=""
    local container_command=""
    local node_host=""
    local node_real=""
    local node_container=""
    local first_line=""
    MOUNT_HOST_COMMAND_RESULT=""

    if [[ -z "${raw_command}" ]]; then
        return 1
    fi

    read -r -a command_parts <<< "${raw_command}"
    executable="${command_parts[0]}"
    if [[ -z "${executable}" ]]; then
        return 1
    fi
    if [[ ${#command_parts[@]} -gt 1 ]]; then
        tail_args="${raw_command#${executable}}"
    fi

    if [[ "${executable}" == /* ]]; then
        if [[ -x "${executable}" ]]; then
            resolved="${executable}"
        else
            return 1
        fi
    else
        resolved="$(command -v "${executable}" || true)"
        if [[ -z "${resolved}" ]]; then
            return 1
        fi
    fi
    resolved_real="$(resolve_real_path "${resolved}")"

    docker_mounts+=(-v "$(dirname "${resolved_real}"):/opt/${mount_name}-host-bin:ro")
    container_command="/opt/${mount_name}-host-bin/$(basename "${resolved_real}")"

    if [[ -r "${resolved_real}" ]]; then
        first_line="$(head -n 1 "${resolved_real}" 2>/dev/null || true)"
        if [[ "${first_line}" == *"node"* ]]; then
            node_host="$(command -v node || true)"
            if [[ -n "${node_host}" ]]; then
                node_real="$(resolve_real_path "${node_host}")"
                docker_mounts+=(-v "$(dirname "${node_real}"):/opt/${mount_name}-node-host-bin:ro")
                node_container="/opt/${mount_name}-node-host-bin/$(basename "${node_real}")"
                container_command="${node_container} ${container_command}"
            fi
        fi
    fi
    if [[ -n "${tail_args}" ]]; then
        container_command="${container_command}${tail_args}"
    fi
    MOUNT_HOST_COMMAND_RESULT="${container_command}"
    return 0
}

for arg in "$@"; do
    case "${arg}" in
        --build)
            BUILD_IMAGE=1
            ;;
        --check)
            CHECK_ONLY=1
            ;;
        --gpus=*)
            GPU_SPEC="${arg#--gpus=}"
            ;;
        --config=*)
            CONFIG_PATH="${arg#--config=}"
            ;;
        --image=*)
            IMAGE="${arg#--image=}"
            ;;
        --host-state-dir=*)
            HOST_STATE_DIR="${arg#--host-state-dir=}"
            ;;
        --state-dir=*)
            CONTAINER_STATE_DIR="${arg#--state-dir=}"
            ;;
        --branch-factor=*)
            BRANCH_FACTOR="${arg#--branch-factor=}"
            ;;
        --max-depth=*)
            MAX_DEPTH="${arg#--max-depth=}"
            ;;
        --max-retries=*)
            MAX_RETRIES="${arg#--max-retries=}"
            ;;
        --docker-user=*)
            DOCKER_USER="${arg#--docker-user=}"
            ;;
        --codex-login)
            CODEX_LOGIN=1
            ;;
        --codex-command=*)
            CODEX_COMMAND_IN_CONTAINER="${arg#--codex-command=}"
            ;;
        --ncu-command=*)
            NCU_COMMAND_IN_CONTAINER="${arg#--ncu-command=}"
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

CONTAINER_WORKSPACE_DIR="${CONTAINER_STATE_DIR}/workspace"
if [[ "${CODEX_LOGIN}" == "1" ]]; then
    CONTAINER_RUNTIME_ROOT="/tmp/autoagent-codex-runtime"
else
    CONTAINER_RUNTIME_ROOT="${CONTAINER_STATE_DIR}/tmp"
fi
CONTAINER_HOME="${CONTAINER_RUNTIME_ROOT}/home"
CONTAINER_TMPDIR="${CONTAINER_RUNTIME_ROOT}/tmp"
CONTAINER_XDG_CONFIG="${CONTAINER_RUNTIME_ROOT}/config"
CONTAINER_XDG_CACHE="${CONTAINER_RUNTIME_ROOT}/cache"
CONTAINER_XDG_DATA="${CONTAINER_RUNTIME_ROOT}/data"
CONTAINER_XDG_STATE="${CONTAINER_RUNTIME_ROOT}/state"
CONTAINER_CODEX_HOME="${CONTAINER_RUNTIME_ROOT}/codex"

AUTH_API_KEY_VALUE=""
AUTH_API_KEY_SOURCE="none"
if [[ -n "${OPENAI_API_KEY_VALUE}" && -n "${CODEX_API_KEY_VALUE}" ]]; then
    echo "Set only one of OPENAI_API_KEY or CODEX_API_KEY before running optimize.sh." >&2
    exit 2
elif [[ -n "${OPENAI_API_KEY_VALUE}" ]]; then
    AUTH_API_KEY_VALUE="${OPENAI_API_KEY_VALUE}"
    AUTH_API_KEY_SOURCE="OPENAI_API_KEY"
elif [[ -n "${CODEX_API_KEY_VALUE}" ]]; then
    AUTH_API_KEY_VALUE="${CODEX_API_KEY_VALUE}"
    AUTH_API_KEY_SOURCE="CODEX_API_KEY"
fi

DOCKER_GPUS="$(normalize_docker_gpus "${GPU_SPEC}")"
HOST_CODEX_BIN="$(command -v codex || true)"
HOST_NODE_BIN="$(command -v node || true)"
RESULTS_CSV="${HOST_STATE_DIR}/kernels/results.csv"

mkdir -p "${HOST_STATE_DIR}"
if [[ "${CODEX_LOGIN}" != "1" ]]; then
    mkdir -p "${HOST_RUNTIME_ROOT}"
    mkdir -p "${HOST_CONTAINER_HOME}"
    mkdir -p "${HOST_CONTAINER_TMP}"
    mkdir -p "${HOST_CONTAINER_XDG_CONFIG}"
    mkdir -p "${HOST_CONTAINER_XDG_CACHE}"
    mkdir -p "${HOST_CONTAINER_XDG_DATA}"
    mkdir -p "${HOST_CONTAINER_XDG_STATE}"
    mkdir -p "${HOST_CONTAINER_CODEX_HOME}"
fi

if [[ "${CODEX_LOGIN}" == "1" ]]; then
    :
elif [[ -n "${AUTH_API_KEY_VALUE}" ]]; then
    :
elif [[ -d "${HOME}/.codex" ]]; then
    (
        cd "${HOME}/.codex"
        tar \
            --exclude='.tmp' \
            --exclude='plugins' \
            --exclude='generated_images' \
            --exclude='logs' \
            --exclude='sessions' \
            -cf - .
    ) | (
        cd "${HOST_CONTAINER_CODEX_HOME}"
        tar -xf -
    )
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for examples/autoagent/optimize.sh" >&2
    exit 2
fi

docker_mounts=(
    -v "${HOST_STATE_DIR}:${CONTAINER_STATE_DIR}"
)

if [[ "${CODEX_LOGIN}" != "1" ]]; then
    docker_mounts+=(-v "${HOST_CONTAINER_CODEX_HOME}:${CONTAINER_CODEX_HOME}")
fi

docker_env=(
    -e HOME="${CONTAINER_HOME}"
    -e TMPDIR="${CONTAINER_TMPDIR}"
    -e XDG_CONFIG_HOME="${CONTAINER_XDG_CONFIG}"
    -e XDG_CACHE_HOME="${CONTAINER_XDG_CACHE}"
    -e XDG_DATA_HOME="${CONTAINER_XDG_DATA}"
    -e XDG_STATE_HOME="${CONTAINER_XDG_STATE}"
    -e CODEX_HOME="${CONTAINER_CODEX_HOME}"
    -e AUTOAGENT_CODEX_RUNTIME_ROOT="${CONTAINER_RUNTIME_ROOT}"
    -e AUTOAGENT_WORKSPACE="${CONTAINER_WORKSPACE_DIR}"
    -e AUTOAGENT_STATE_DIR="${CONTAINER_STATE_DIR}"
    -e AUTOAGENT_PROFILE_EXECUTOR="${PROFILE_EXECUTOR}"
    -e AUTOAGENT_NCU_COMMAND="${NCU_COMMAND_IN_CONTAINER}"
)

if [[ "${CODEX_LOGIN}" == "1" ]]; then
    :
elif [[ -n "${AUTH_API_KEY_VALUE}" ]]; then
    docker_env+=(
        -e OPENAI_API_KEY="${AUTH_API_KEY_VALUE}"
        -e CODEX_API_KEY="${AUTH_API_KEY_VALUE}"
        -e CODEX_ACCESS_TOKEN=
    )
fi

if [[ -n "${CODEX_COMMAND_IN_CONTAINER}" ]]; then
    if mount_host_command_if_needed "${CODEX_COMMAND_IN_CONTAINER}" "autoagent-codex"; then
        CODEX_COMMAND_IN_CONTAINER="${MOUNT_HOST_COMMAND_RESULT}"
    fi
    docker_env+=(-e AUTOAGENT_CODEX_COMMAND="${CODEX_COMMAND_IN_CONTAINER}")
fi

if [[ "${BUILD_IMAGE}" == "1" ]]; then
    echo "Building image: ${IMAGE}"
    docker build -t "${IMAGE}" "${EXAMPLE_DIR}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Docker image not found: ${IMAGE}" >&2
    echo "Re-run with --build or pass --image=<existing-tag>." >&2
    exit 2
fi

echo "Host preflight:"
echo "  Image: ${IMAGE}"
echo "  Config: ${CONFIG_PATH}"
echo "  Host state dir: ${HOST_STATE_DIR}"
echo "  Container state dir: ${CONTAINER_STATE_DIR}"
echo "  Container workspace dir: ${CONTAINER_WORKSPACE_DIR}"
echo "  Host runtime root: ${HOST_RUNTIME_ROOT}"
echo "  Host codex stage root: ${HOST_CODEX_STAGE_ROOT}"
echo "  Host codex stage dir: ${HOST_CODEX_STAGE_DIR}"
echo "  Container runtime root: ${CONTAINER_RUNTIME_ROOT}"
echo "  Host container HOME: ${HOST_CONTAINER_HOME}"
echo "  Container HOME: ${CONTAINER_HOME}"
echo "  Host TMPDIR: ${HOST_CONTAINER_TMP}"
echo "  Container TMPDIR: ${CONTAINER_TMPDIR}"
echo "  Container XDG_CONFIG_HOME: ${CONTAINER_XDG_CONFIG}"
echo "  Container XDG_CACHE_HOME: ${CONTAINER_XDG_CACHE}"
echo "  Container XDG_DATA_HOME: ${CONTAINER_XDG_DATA}"
echo "  Container XDG_STATE_HOME: ${CONTAINER_XDG_STATE}"
echo "  Container CODEX_HOME: ${CONTAINER_CODEX_HOME}"
echo "  Docker user: ${DOCKER_USER}"
echo "  Docker GPUs: ${DOCKER_GPUS}"
echo "  Max depth: ${MAX_DEPTH}"
echo "  Branch factor: ${BRANCH_FACTOR}"
echo "  Max retries: ${MAX_RETRIES}"
echo "  Codex login mode: ${CODEX_LOGIN}"
echo "  Host codex binary: ${HOST_CODEX_BIN:-<not found>}"
echo "  Host node binary: ${HOST_NODE_BIN:-<not found>}"
echo "  Host codex auth dir: ${HOME}/.codex"
if [[ "${CODEX_LOGIN}" == "1" ]]; then
    echo "  OPENAI_API_KEY: not used (temporary container login)"
elif [[ -n "${AUTH_API_KEY_VALUE}" ]]; then
    echo "  Auth env source: ${AUTH_API_KEY_SOURCE}"
else
    echo "  Auth env source: none"
fi
if [[ -n "${CODEX_COMMAND_IN_CONTAINER}" ]]; then
    echo "  Container codex command: ${CODEX_COMMAND_IN_CONTAINER}"
else
    echo "  Container codex command: codex (from image PATH)"
fi
echo "  Container ncu command: ${NCU_COMMAND_IN_CONTAINER}"

if [[ "${CODEX_LOGIN}" != "1" && "${MAX_DEPTH}" != "0" && -z "${AUTH_API_KEY_VALUE}" && ! -d "${HOME}/.codex" ]]; then
    echo "Warning: ${HOME}/.codex was not found. Codex CLI authentication may fail inside the container." >&2
fi

docker_run_args=(
    run
    --rm
    --gpus "${DOCKER_GPUS}"
    --cap-add SYS_ADMIN
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --ipc host
    --user "${DOCKER_USER}"
    --workdir "${CONTAINER_STATE_DIR}"
    "${docker_mounts[@]}"
    "${docker_env[@]}"
)

docker_image_args=(
    "${IMAGE}"
)

stage_workspace_script="set -eu
rm -rf \"${CONTAINER_WORKSPACE_DIR}\"
mkdir -p \"${CONTAINER_WORKSPACE_DIR}\"
(cd /app && tar --exclude=.venv -cf - .) | (cd \"${CONTAINER_WORKSPACE_DIR}\" && tar -xf -)
mkdir -p \"${CONTAINER_TMPDIR}\" \"${CONTAINER_XDG_CONFIG}\" \"${CONTAINER_XDG_CACHE}\" \"${CONTAINER_XDG_DATA}\" \"${CONTAINER_XDG_STATE}\" \"${CONTAINER_HOME}\" \"${CONTAINER_CODEX_HOME}\"
"

echo
echo "Staging writable workspace from image..."
docker "${docker_run_args[@]}" "${docker_image_args[@]}" /bin/sh -lc "${stage_workspace_script}"

echo
echo "Container capability check:"
printf '  Command:'
printf ' %q' /app/.venv/bin/python "${CONTAINER_WORKSPACE_DIR}/main.py" \
    --config "${CONFIG_PATH}" \
    --workspace "${CONTAINER_WORKSPACE_DIR}" \
    --state-dir "${CONTAINER_STATE_DIR}" \
    --backend codex-exec \
    --profile-executor "${PROFILE_EXECUTOR}" \
    --branch-factor "${BRANCH_FACTOR}" \
    --max-depth "${MAX_DEPTH}" \
    --max-retries "${MAX_RETRIES}" \
    --print-capabilities
printf '\n'
docker "${docker_run_args[@]}" \
    "${docker_image_args[@]}" \
    /app/.venv/bin/python "${CONTAINER_WORKSPACE_DIR}/main.py" \
    --config "${CONFIG_PATH}" \
    --workspace "${CONTAINER_WORKSPACE_DIR}" \
    --state-dir "${CONTAINER_STATE_DIR}" \
    --backend codex-exec \
    --profile-executor "${PROFILE_EXECUTOR}" \
    --branch-factor "${BRANCH_FACTOR}" \
    --max-depth "${MAX_DEPTH}" \
    --max-retries "${MAX_RETRIES}" \
    --print-capabilities

if [[ "${MAX_DEPTH}" != "0" && -n "${CODEX_COMMAND_IN_CONTAINER}" ]]; then
    echo
    echo "Container codex preflight:"
    printf '  Command:'
    printf ' %q' /bin/sh -lc "${CODEX_COMMAND_IN_CONTAINER} --version"
    printf '\n'
    docker "${docker_run_args[@]}" "${docker_image_args[@]}" /bin/sh -lc "${CODEX_COMMAND_IN_CONTAINER} --version"
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
    exit 0
fi

echo
echo "Running optimize flow..."
if [[ "${CODEX_LOGIN}" == "1" && "${MAX_DEPTH}" != "0" ]]; then
    echo "  Action: temporary in-container 'codex login --with-api-key'"
    echo "  Input: paste the API key when prompted, then press Ctrl-D"
    docker "${docker_run_args[@]}" -i "${docker_image_args[@]}" /bin/sh -lc "
set -eu
mkdir -p \"${CONTAINER_TMPDIR}\" \"${CONTAINER_XDG_CONFIG}\" \"${CONTAINER_XDG_CACHE}\" \"${CONTAINER_XDG_DATA}\" \"${CONTAINER_XDG_STATE}\" \"${CONTAINER_HOME}\" \"${CONTAINER_CODEX_HOME}\"
echo 'Temporary Codex login inside container runtime...'
${CODEX_COMMAND_IN_CONTAINER:-codex} login --with-api-key
/app/.venv/bin/python \"${CONTAINER_WORKSPACE_DIR}/main.py\" \
  --config \"${CONFIG_PATH}\" \
  --workspace \"${CONTAINER_WORKSPACE_DIR}\" \
  --state-dir \"${CONTAINER_STATE_DIR}\" \
  --backend codex-exec \
  --profile-executor \"${PROFILE_EXECUTOR}\" \
  --branch-factor \"${BRANCH_FACTOR}\" \
  --max-depth \"${MAX_DEPTH}\" \
  --max-retries \"${MAX_RETRIES}\" \
  --optimize
"
else
    printf '  Command:'
    printf ' %q' /app/.venv/bin/python "${CONTAINER_WORKSPACE_DIR}/main.py" \
        --config "${CONFIG_PATH}" \
        --workspace "${CONTAINER_WORKSPACE_DIR}" \
        --state-dir "${CONTAINER_STATE_DIR}" \
        --backend codex-exec \
        --profile-executor "${PROFILE_EXECUTOR}" \
        --branch-factor "${BRANCH_FACTOR}" \
        --max-depth "${MAX_DEPTH}" \
        --max-retries "${MAX_RETRIES}" \
        --optimize
    printf '\n'
    docker "${docker_run_args[@]}" \
        "${docker_image_args[@]}" \
        /app/.venv/bin/python "${CONTAINER_WORKSPACE_DIR}/main.py" \
        --config "${CONFIG_PATH}" \
        --workspace "${CONTAINER_WORKSPACE_DIR}" \
        --state-dir "${CONTAINER_STATE_DIR}" \
        --backend codex-exec \
        --profile-executor "${PROFILE_EXECUTOR}" \
        --branch-factor "${BRANCH_FACTOR}" \
        --max-depth "${MAX_DEPTH}" \
        --max-retries "${MAX_RETRIES}" \
        --optimize
fi

echo
echo "Expected results:"
echo "  ${RESULTS_CSV}"
