#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENSHELL_BIN="${OPENSHELL_BIN:-openshell}"
EXAMPLE_DIR="${KDA_EXAMPLE_DIR:-${ROOT}/examples/kernel-design-agents}"
POLICY_FILE="${KDA_SANDBOX_POLICY:-${EXAMPLE_DIR}/sandbox-policy.yaml}"
SANDBOX_NAME="${KDA_SANDBOX_NAME:-kda-smoke-$(date +%s)-$$}"
KEEP_SANDBOX="${KDA_KEEP_SANDBOX:-0}"
GPU_DEVICE="${KDA_GPU_DEVICE:-}"

usage() {
    cat <<'EOF'
Usage: bash examples/kernel-design-agents/smoke.sh [options]

Options:
  --gpus=<id>      Shorthand for OpenShell GPU selection.
  --name=<name>    Override sandbox name.
  --keep           Preserve the sandbox after the script exits.
  -h, --help       Show this help.
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

for arg in "$@"; do
    case "${arg}" in
        --gpus=*)
            GPU_DEVICE="$(normalize_gpu_device "${arg#--gpus=}")"
            ;;
        --name=*)
            SANDBOX_NAME="${arg#--name=}"
            ;;
        --keep)
            KEEP_SANDBOX=1
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

cleanup() {
    local exit_code=$?

    if [[ "${KEEP_SANDBOX}" == "1" ]]; then
        echo "Preserving sandbox: ${SANDBOX_NAME}"
        exit "${exit_code}"
    fi

    "${OPENSHELL_BIN}" sandbox delete "${SANDBOX_NAME}" >/dev/null 2>&1 || true
    exit "${exit_code}"
}
trap cleanup EXIT

if ! "${OPENSHELL_BIN}" status >/dev/null 2>&1; then
    echo "No healthy OpenShell gateway is reachable." >&2
    exit 2
fi

create_args=(
    sandbox create
    --name "${SANDBOX_NAME}"
    --keep
    --no-tty
    --no-auto-providers
    --from "${EXAMPLE_DIR}"
    --gpu
    --policy "${POLICY_FILE}"
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

run_exec() {
    local description="$1"
    shift
    echo
    echo "== ${description} =="
    "${OPENSHELL_BIN}" sandbox exec -n "${SANDBOX_NAME}" -- "$@"
}

run_exec \
    "Verify packaged tools" \
    /bin/sh -lc 'runtime_root=/tmp/kda-smoke-codex-runtime && mkdir -p "$runtime_root/home" "$runtime_root/tmp" "$runtime_root/cache" "$runtime_root/config" "$runtime_root/data" "$runtime_root/state" "$runtime_root/codex" "$runtime_root/codex/sqlite" && export HOME="$runtime_root/home" TMPDIR="$runtime_root/tmp" TMP="$runtime_root/tmp" TEMP="$runtime_root/tmp" XDG_CACHE_HOME="$runtime_root/cache" XDG_CONFIG_HOME="$runtime_root/config" XDG_DATA_HOME="$runtime_root/data" XDG_STATE_HOME="$runtime_root/state" CODEX_HOME="$runtime_root/codex" CODEX_SQLITE_HOME="$runtime_root/codex/sqlite" && codex --version && node --version && git --version && jq --version'

run_exec \
    "Verify scaffold script" \
    /bin/bash -lc "rm -rf /tmp/kda-smoke && /app/scaffold.sh /tmp/kda-smoke && test -f /tmp/kda-smoke/TASK_CONTRACT.md && test -f /tmp/kda-smoke/config.toml && test -f /tmp/kda-smoke/solution/triton/kernel.py && test -f /tmp/kda-smoke/scripts/run_local.py && test -f /tmp/kda-smoke/benchmark.csv"

run_exec \
    "Verify draft runner help" \
    /bin/bash /app/scripts/run-kda-draft.sh --help

echo
echo "Kernel Design Agents smoke passed in sandbox: ${SANDBOX_NAME}"
