#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="${KDA_EXAMPLE_DIR:-${ROOT}/examples/kernel-design-agents}"
OPENSHELL_BIN="${OPENSHELL_BIN:-openshell}"
POLICY_FILE="${KDA_SANDBOX_POLICY:-${EXAMPLE_DIR}/sandbox-policy.yaml}"
WORKSPACE=""
SANDBOX_NAME="${KDA_SANDBOX_NAME:-kda-plan-$(date +%s)-$$}"
PROVIDER_NAME="${KDA_PROVIDER_NAME:-kda-codex-$(date +%s)-$$}"
KEEP_SANDBOX=0
GPU_DEVICE="${KDA_GPU_DEVICE:-}"
MODEL="${KDA_CODEX_MODEL:-gpt-5.4-mini}"
REASONING="${KDA_CODEX_REASONING:-low}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/openshell-kda-plan.XXXXXX")"
OPENAI_API_KEY_VALUE="${OPENAI_API_KEY:-}"
CODEX_API_KEY_VALUE="${CODEX_API_KEY:-}"
AUTH_MODE="oauth"

PROVIDER_CREATED=0
SANDBOX_CREATED=0

usage() {
    cat <<'EOF'
Usage: bash examples/kernel-design-agents/plan.sh --workspace <path> [options]

Options:
  --workspace <path>  Local task workspace to upload and later download back.
  --gpus=<id>         Shorthand for `--gpu-device nvidia.com/gpu=<id>`.
  --name=<name>       Override sandbox name.
  --model=<name>      Codex model override.
  --reasoning=<lvl>   Codex reasoning effort. Default: low
  --keep              Preserve the sandbox and provider after the run.
  -h, --help          Show this help.
EOF
}

normalize_gpu_device() {
    local value="$1"
    if [[ "${value}" == *"="* || "${value}" == *"/"* ]]; then
        printf '%s\n' "${value}"
        return 0
    fi
    printf 'nvidia.com/gpu=%s\n' "${value}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --gpus=*)
            GPU_DEVICE="$(normalize_gpu_device "${1#--gpus=}")"
            shift
            ;;
        --name=*)
            SANDBOX_NAME="${1#--name=}"
            shift
            ;;
        --model=*)
            MODEL="${1#--model=}"
            shift
            ;;
        --reasoning=*)
            REASONING="${1#--reasoning=}"
            shift
            ;;
        --keep)
            KEEP_SANDBOX=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cleanup() {
    local status=$?

    if [[ "${KEEP_SANDBOX}" == "1" ]]; then
        echo "Preserving sandbox: ${SANDBOX_NAME}"
        echo "Preserving provider: ${PROVIDER_NAME}"
        echo "Temporary files kept at: ${TMP_DIR}"
        exit "${status}"
    fi

    if [[ "${SANDBOX_CREATED}" == "1" ]]; then
        "${OPENSHELL_BIN}" sandbox delete "${SANDBOX_NAME}" >/dev/null 2>&1 || true
    fi
    if [[ "${PROVIDER_CREATED}" == "1" ]]; then
        "${OPENSHELL_BIN}" provider delete "${PROVIDER_NAME}" >/dev/null 2>&1 || true
    fi
    rm -rf "${TMP_DIR}"
    exit "${status}"
}
trap cleanup EXIT

if [[ -z "${WORKSPACE}" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -d "${WORKSPACE}" ]]; then
    echo "Workspace does not exist: ${WORKSPACE}" >&2
    exit 2
fi

if [[ ! -f "${WORKSPACE}/TASK_CONTRACT.md" ]]; then
    echo "Workspace is missing TASK_CONTRACT.md: ${WORKSPACE}" >&2
    exit 2
fi

if [[ -n "${OPENAI_API_KEY_VALUE}" && -n "${CODEX_API_KEY_VALUE}" ]]; then
    echo "Set only one of OPENAI_API_KEY or CODEX_API_KEY before running plan.sh." >&2
    exit 2
elif [[ -n "${OPENAI_API_KEY_VALUE}" || -n "${CODEX_API_KEY_VALUE}" ]]; then
    AUTH_MODE="api-key"
    if [[ -z "${OPENAI_API_KEY_VALUE}" ]]; then
        OPENAI_API_KEY_VALUE="${CODEX_API_KEY_VALUE}"
    fi
    if [[ -z "${CODEX_API_KEY_VALUE}" ]]; then
        CODEX_API_KEY_VALUE="${OPENAI_API_KEY_VALUE}"
    fi
    export OPENAI_API_KEY="${OPENAI_API_KEY_VALUE}"
    export CODEX_API_KEY="${CODEX_API_KEY_VALUE}"
fi

if [[ "${AUTH_MODE}" == "oauth" ]]; then
    if ! command -v jq >/dev/null 2>&1; then
        echo "jq is required for ${0}" >&2
        exit 2
    fi
    if [[ ! -f "${HOME}/.codex/auth.json" ]]; then
        echo "Missing local Codex sign-in. Run: codex login, or set OPENAI_API_KEY." >&2
        exit 2
    fi
fi

if ! "${OPENSHELL_BIN}" status >/dev/null 2>&1; then
    echo "No healthy OpenShell gateway is reachable." >&2
    exit 2
fi

echo "Creating provider: ${PROVIDER_NAME}"
if [[ "${AUTH_MODE}" == "api-key" ]]; then
    "${OPENSHELL_BIN}" provider create \
        --name "${PROVIDER_NAME}" \
        --type generic \
        --credential OPENAI_API_KEY \
        --credential CODEX_API_KEY >/dev/null
else
    CODEX_AUTH_ACCESS_TOKEN="$(jq -r '.tokens.access_token // empty' "${HOME}/.codex/auth.json")"
    CODEX_AUTH_REFRESH_TOKEN="$(jq -r '.tokens.refresh_token // empty' "${HOME}/.codex/auth.json")"
    CODEX_AUTH_ACCOUNT_ID="$(jq -r '.tokens.account_id // empty' "${HOME}/.codex/auth.json")"
    export CODEX_AUTH_ACCESS_TOKEN CODEX_AUTH_REFRESH_TOKEN CODEX_AUTH_ACCOUNT_ID

    if [[ -z "${CODEX_AUTH_ACCESS_TOKEN}" || -z "${CODEX_AUTH_REFRESH_TOKEN}" || -z "${CODEX_AUTH_ACCOUNT_ID}" ]]; then
        echo "Local Codex auth.json is missing one or more required fields." >&2
        exit 2
    fi

    "${OPENSHELL_BIN}" provider create \
        --name "${PROVIDER_NAME}" \
        --type generic \
        --credential CODEX_AUTH_ACCESS_TOKEN \
        --credential CODEX_AUTH_REFRESH_TOKEN \
        --credential CODEX_AUTH_ACCOUNT_ID >/dev/null
fi
PROVIDER_CREATED=1

create_args=(
    sandbox create
    --name "${SANDBOX_NAME}"
    --keep
    --no-tty
    --no-auto-providers
    --from "${EXAMPLE_DIR}"
    --gpu
    --policy "${POLICY_FILE}"
    --provider "${PROVIDER_NAME}"
)

if [[ -n "${GPU_DEVICE}" ]]; then
    create_args+=(--gpu-device "${GPU_DEVICE}")
fi

create_args+=(
    --
    /bin/sh -lc "echo kda sandbox ready"
)

echo "Creating sandbox: ${SANDBOX_NAME}"
"${OPENSHELL_BIN}" "${create_args[@]}"
SANDBOX_CREATED=1

REMOTE_WORKSPACE_PARENT="/sandbox"
REMOTE_WORKSPACE="${REMOTE_WORKSPACE_PARENT}/$(basename "${WORKSPACE}")"
DOWNLOAD_ROOT="${TMP_DIR}/download"
mkdir -p "${DOWNLOAD_ROOT}"

echo "Uploading workspace to sandbox: ${REMOTE_WORKSPACE}"
"${OPENSHELL_BIN}" sandbox upload "${SANDBOX_NAME}" "${WORKSPACE}" "${REMOTE_WORKSPACE_PARENT}"

echo "Running KDA draft loop in sandbox"
"${OPENSHELL_BIN}" sandbox exec -n "${SANDBOX_NAME}" -- \
    /bin/bash /app/scripts/run-kda-draft.sh \
    --workspace "${REMOTE_WORKSPACE}" \
    --model "${MODEL}" \
    --reasoning "${REASONING}"

echo "Downloading updated workspace back to host"
"${OPENSHELL_BIN}" sandbox download "${SANDBOX_NAME}" "${REMOTE_WORKSPACE}" "${DOWNLOAD_ROOT}"
cp -R "${DOWNLOAD_ROOT}/$(basename "${WORKSPACE}")/." "${WORKSPACE}/"

echo
echo "KDA draft completed."
echo "Workspace: ${WORKSPACE}"
echo "Draft: ${WORKSPACE}/docs/draft.md"
echo "Last message: ${WORKSPACE}/outputs/last-message.md"
