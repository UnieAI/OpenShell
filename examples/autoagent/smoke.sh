#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENSHELL_BIN="${OPENSHELL_BIN:-openshell}"
EXAMPLE_DIR="${AUTOAGENT_EXAMPLE_DIR:-${ROOT}/examples/autoagent}"
POLICY_FILE="${AUTOAGENT_SANDBOX_POLICY:-${EXAMPLE_DIR}/sandbox-policy.yaml}"
CONFIG_PATH="${AUTOAGENT_CONFIG:-configs/gemm.yaml}"
STATE_DIR="${AUTOAGENT_STATE_DIR:-/sandbox/autoagent}"
SANDBOX_NAME="${AUTOAGENT_SANDBOX_NAME:-autoagent-smoke-$(date +%s)-$$}"
KEEP_SANDBOX="${AUTOAGENT_KEEP_SANDBOX:-0}"
GPU_DEVICE="${AUTOAGENT_GPU_DEVICE:-}"

usage() {
    cat <<'EOF'
Usage: bash examples/autoagent/smoke.sh [options]

Options:
  --gpus=<id>      Shorthand for OpenShell GPU selection. `--gpus=1` maps to
                   `--gpu-device nvidia.com/gpu=1`.
  --name=<name>    Override sandbox name.
  --config=<path>  Override kernel config path inside the image workspace.
  --state-dir=<p>  Override writable state dir inside the sandbox.
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
        --config=*)
            CONFIG_PATH="${arg#--config=}"
            ;;
        --state-dir=*)
            STATE_DIR="${arg#--state-dir=}"
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
    --from "${EXAMPLE_DIR}"
    --gpu
    --policy "${POLICY_FILE}"
)

if [[ -n "${GPU_DEVICE}" ]]; then
    create_args+=(--gpu-device "${GPU_DEVICE}")
fi

create_args+=(
    --
    /bin/sh -lc "echo autoagent sandbox ready"
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
    "Verify packaged Python dependencies" \
    /app/.venv/bin/python -c \
    "import sys, torch, tvm_ffi; print(sys.executable); print(torch.__version__); print(tvm_ffi.__file__)"

run_exec \
    "Compile and verify GEMM" \
    /app/.venv/bin/python /app/main.py \
    --workspace /app \
    --state-dir "${STATE_DIR}" \
    --config "${CONFIG_PATH}"

run_exec \
    "Verify-only rerun" \
    /app/.venv/bin/python /app/main.py \
    --workspace /app \
    --state-dir "${STATE_DIR}" \
    --config "${CONFIG_PATH}" \
    --test-only

run_exec \
    "Check expected artifacts" \
    /bin/sh -lc "test -f '${STATE_DIR}/kernel.so' && test -f '${STATE_DIR}/active_kernel.yaml'"

echo
echo "AutoAgent smoke passed in sandbox: ${SANDBOX_NAME}"
