#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE=""
MODEL="${KDA_CODEX_MODEL:-gpt-5.4-mini}"
REASONING="${KDA_CODEX_REASONING:-low}"
PROMPT_FILE="/app/prompts/basic-flow.md"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_RUNTIME_ROOT="${KDA_CODEX_RUNTIME_ROOT:-}"
MODE="${KDA_EXECUTION_MODE:-draft}"
KERNEL_OPTIMIZE="${KDA_KERNEL_OPTIMIZE:-0}"
KERNEL_OPTIMIZE_STATE="disabled"
MAX_DEPTH="${KDA_MAX_DEPTH:-1}"
PRE_EXECUTION_SUMMARY_HASH=""

usage() {
    cat <<'EOF'
Usage: /bin/bash /app/scripts/run-kda-draft.sh --workspace <path> [options]

Options:
  --workspace <path>   Uploaded task workspace inside the sandbox.
  --model <name>       Codex model override.
  --reasoning <level>  Codex reasoning effort.
  --mode <name>        `draft` or `execute`. Default: draft
  --prompt-file <path> Prompt template to use. Default: /app/prompts/basic-flow.md
  --runtime-root <p>   Writable Codex runtime root. Default: <workspace>/outputs/codex-runtime
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --reasoning)
            REASONING="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --runtime-root)
            CODEX_RUNTIME_ROOT="$2"
            shift 2
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

case "${MODE}" in
    draft|execute)
        ;;
    *)
        echo "Unsupported mode: ${MODE}. Expected draft or execute." >&2
        exit 2
        ;;
esac

if [[ -z "${WORKSPACE}" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -d "${WORKSPACE}" ]]; then
    echo "Workspace does not exist: ${WORKSPACE}" >&2
    exit 2
fi

if [[ ! -f "${WORKSPACE}/TASK_CONTRACT.md" ]]; then
    echo "Missing TASK_CONTRACT.md in workspace: ${WORKSPACE}" >&2
    exit 2
fi

mkdir -p "${WORKSPACE}/docs" "${WORKSPACE}/runs" "${WORKSPACE}/outputs" "${WORKSPACE}/profile"
touch "${WORKSPACE}/candidates.jsonl"
if [[ ! -f "${WORKSPACE}/benchmark.csv" ]]; then
    printf 'candidate,metric,unit,status,notes\n' > "${WORKSPACE}/benchmark.csv"
fi
if [[ "${MODE}" == "execute" ]]; then
    cat > "${WORKSPACE}/outputs/execution-summary.md" <<'EOF'
# Execution Summary

The agent updates this file after implementation, validation, and optional evaluation.
EOF
    PRE_EXECUTION_SUMMARY_HASH="$(cksum "${WORKSPACE}/outputs/execution-summary.md" | awk '{print $1 ":" $2}')"
fi

setup_codex_runtime_env() {
    local runtime_root home_dir tmp_dir xdg_cache_dir xdg_config_dir xdg_data_dir xdg_state_dir codex_home_dir codex_sqlite_dir codex_logs_dir

    if [[ -z "${CODEX_RUNTIME_ROOT}" ]]; then
        CODEX_RUNTIME_ROOT="${WORKSPACE}/outputs/codex-runtime"
    fi

    runtime_root="${CODEX_RUNTIME_ROOT}"
    home_dir="${runtime_root}/home"
    tmp_dir="${runtime_root}/tmp"
    xdg_cache_dir="${runtime_root}/cache"
    xdg_config_dir="${runtime_root}/config"
    xdg_data_dir="${runtime_root}/data"
    xdg_state_dir="${runtime_root}/state"
    codex_home_dir="${runtime_root}/codex"
    codex_sqlite_dir="${codex_home_dir}/sqlite"
    codex_logs_dir="${codex_home_dir}/logs"

    mkdir -p \
        "${runtime_root}" \
        "${home_dir}" \
        "${tmp_dir}" \
        "${xdg_cache_dir}" \
        "${xdg_config_dir}" \
        "${xdg_data_dir}" \
        "${xdg_state_dir}" \
        "${codex_home_dir}" \
        "${codex_sqlite_dir}" \
        "${codex_logs_dir}"

    export HOME="${home_dir}"
    export TMPDIR="${tmp_dir}"
    export TMP="${tmp_dir}"
    export TEMP="${tmp_dir}"
    export XDG_CACHE_HOME="${xdg_cache_dir}"
    export XDG_CONFIG_HOME="${xdg_config_dir}"
    export XDG_DATA_HOME="${xdg_data_dir}"
    export XDG_STATE_HOME="${xdg_state_dir}"
    export CODEX_HOME="${codex_home_dir}"
    export CODEX_SQLITE_HOME="${codex_sqlite_dir}"
    export KDA_CODEX_LOG_DIR="${codex_logs_dir}"
}

setup_codex_runtime_env

workspace_has_validated_baseline() {
    local candidates_file="$1"
    local results_file="$2"

    if [[ -s "${candidates_file}" ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"(validated|passed)"' "${candidates_file}"; then
        return 0
    fi

    if [[ -s "${results_file}" ]]; then
        python3 - "${results_file}" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    data = json.load(open(path))
except Exception:
    raise SystemExit(1)

results = data.get("results", {})
seen = False
for definition_results in results.values():
    if not isinstance(definition_results, dict):
        continue
    for workload_result in definition_results.values():
        if not isinstance(workload_result, dict):
            continue
        seen = True
        if workload_result.get("status") != "PASSED":
            raise SystemExit(1)

if not seen:
    raise SystemExit(1)
PY
        return $?
    fi

    return 1
}

bootstrap_codex_oauth() {
    mkdir -p "${HOME}/.codex"
    node - <<'NODE'
const fs = require("fs");
const path = `${process.env.HOME}/.codex/auth.json`;
const b64u = (obj) => Buffer.from(JSON.stringify(obj)).toString("base64url");
const now = Math.floor(Date.now() / 1000);
const fakeIdToken = [
  b64u({ alg: "none", typ: "JWT" }),
  b64u({
    iss: "https://auth.openai.com",
    aud: "codex",
    sub: "openshell-kda",
    email: "kda@openshell.local",
    iat: now,
    exp: now + 3600,
  }),
  "placeholder",
].join(".");

fs.writeFileSync(path, JSON.stringify({
  auth_mode: "chatgpt",
  OPENAI_API_KEY: null,
  tokens: {
    id_token: fakeIdToken,
    access_token: process.env.CODEX_AUTH_ACCESS_TOKEN,
    refresh_token: process.env.CODEX_AUTH_REFRESH_TOKEN,
    account_id: process.env.CODEX_AUTH_ACCOUNT_ID,
  },
  last_refresh: new Date().toISOString(),
}, null, 2));
NODE
    chmod 600 "${HOME}/.codex/auth.json"
}

if [[ -n "${CODEX_AUTH_ACCESS_TOKEN:-}" && -n "${CODEX_AUTH_REFRESH_TOKEN:-}" && -n "${CODEX_AUTH_ACCOUNT_ID:-}" ]]; then
    bootstrap_codex_oauth
elif [[ -z "${OPENAI_API_KEY:-}" && -z "${CODEX_API_KEY:-}" ]]; then
    echo "No Codex auth material found in the sandbox environment." >&2
    echo "Expected CODEX_AUTH_ACCESS_TOKEN/CODEX_AUTH_REFRESH_TOKEN/CODEX_AUTH_ACCOUNT_ID or OPENAI_API_KEY/CODEX_API_KEY." >&2
    exit 2
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Prompt file does not exist: ${PROMPT_FILE}" >&2
    exit 2
fi

build_workspace_inspection_block() {
    local candidate
    local -a candidates=(
        "README.md"
        "FAQ.md"
        "EVALUATION.md"
        "TASK_CONTRACT.md"
        "config.toml"
        "scripts/check_cuda_extension.py"
        "scripts/pack_solution.py"
        "scripts/run_local.py"
        "scripts/run_modal.py"
        "solution/triton/kernel.py"
        "solution/cuda/kernel.cu"
        "solution/cuda/binding.py"
        "docs/draft.md"
        "docs/plan.md"
    )

    printf '\n## Required Workspace Inspection\n\n'
    printf 'Before drafting, read the following files if they exist in the workspace. Use them to justify baseline claims, the active implementation path, and the validation/evaluation commands.\n\n'

    for candidate in "${candidates[@]}"; do
        if [[ -f "${WORKSPACE}/${candidate}" ]]; then
            printf -- '- `%s`\n' "${candidate}"
        fi
    done

    printf '\nDo not claim facts about any of those files unless you actually read them in this run.\n'
}

build_execution_mode_block() {
    if [[ "${MODE}" == "draft" ]]; then
        cat <<EOF

## Execution Mode

This run is draft-only.

You must:

1. Read \`TASK_CONTRACT.md\`.
2. Inspect the local workspace before proposing changes.
3. Write the first implementation-plan draft to \`docs/draft.md\`.
4. Stop after the draft exists.

Do not implement the task in this run.
Do not edit files outside the workspace root.
EOF
        return 0
    fi

    cat <<EOF

## Execution Mode

This run is implementation plus validation.

Environment:

- Workspace root: ${WORKSPACE}
- FIB_DATASET_PATH: ${FIB_DATASET_PATH:-<unset>}
- Draft file: ${WORKSPACE}/docs/draft.md
- Plan file: ${WORKSPACE}/docs/plan.md
- Summary file: ${WORKSPACE}/outputs/execution-summary.md
- Candidate log: ${WORKSPACE}/candidates.jsonl
- Benchmark table: ${WORKSPACE}/benchmark.csv
- Run artifacts directory: ${WORKSPACE}/runs
- Canonical benchmark log path: ${WORKSPACE}/runs/run_local.txt
- Canonical benchmark results path: ${WORKSPACE}/runs/run_local_results.json

Submission rules:

- Treat local dataset definitions, workloads, and baseline solutions as semantics and benchmarking references only.
- Do not call runtime APIs from specialized external kernel libraries such as \`flashinfer\` or \`deep_gemm\` unless \`TASK_CONTRACT.md\` explicitly allows it.
- Keep the candidate self-contained under \`solution/\` and aligned with the active \`config.toml\` build path.
- If the active build is CUDA, keep the implementation surface aligned with the configured \`binding\` mode instead of mixing TVM FFI and Torch-extension patterns.
- If the active build is CUDA with \`binding=torch\`, keep \`config.toml\` on a CUDA entry point such as \`kernel.cu::kernel\`. Do not switch the active benchmark path to \`binding.py::...\`.
- Use staged validation for expensive kernels: pack first, then any applicable direct compile smoke, then the reduced benchmark subset configured in the environment, and only then attempt a broader sweep if the candidate still looks viable.
- If \`scripts/check_cuda_extension.py\` exists and the active build is CUDA with \`binding=torch\`, use it for the compile-smoke stage before running \`scripts/run_local.py\`.
- When you run \`scripts/run_local.py\`, preserve the benchmark artifacts at \`runs/run_local.txt\` and \`runs/run_local_results.json\`. Do not invent alternate filenames.
- Do not count edits to \`scripts/pack_solution.py\`, \`scripts/run_local.py\`, or docs-only files as kernel progress unless the contract explicitly asks for tooling work.
- If the candidate returns tensors instead of writing to preallocated outputs, align \`destination_passing_style\` with the actual callable signature.

EOF

    if [[ "${KERNEL_OPTIMIZE}" == "1" ]]; then
        if [[ "${KERNEL_OPTIMIZE_STATE}" == "bootstrap" ]]; then
            cat <<EOF

## Kernel Optimization Focus

This run is explicitly for kernel optimization, but the workspace does not yet
have a validated baseline candidate.

- First bootstrap one correct baseline candidate and validate it on the active
  workload subset.
- After that baseline is validated, continue in the same run to attempt up to
  ${MAX_DEPTH} optimization candidate(s) if no concrete blocker prevents it.
- Treat the freshly validated baseline as the parent candidate for the first
  optimization attempt.
- Prioritize edits under \`solution/\`, especially the active CUDA or Triton
  implementation files.
- Avoid spending the run on scaffold churn, config rewrites, prompt-only edits,
  or alternate binding experiments unless a concrete blocker forces it.
- Preserve the active build path, callable signature, and validation flow while
  improving performance.
EOF
        else
            cat <<EOF

## Kernel Optimization Focus

This run is explicitly for kernel optimization, not baseline bring-up.

- Start from the current validated candidate and improve latency on the active workload subset.
- Attempt up to ${MAX_DEPTH} optimization candidate(s) in this run.
- Prioritize edits under \`solution/\`, especially the active CUDA or Triton implementation files.
- Avoid spending the run on scaffold churn, config rewrites, prompt-only edits, or alternate binding experiments unless a concrete blocker forces it.
- Preserve the active build path, callable signature, and validation flow while improving performance.
- Use existing \`runs/run_local_results.json\`, \`benchmark.csv\`, and \`candidates.jsonl\` as the before-state when available, then record the new candidate against that baseline.
EOF
        fi
    fi

    cat <<EOF

You must:

1. Read \`TASK_CONTRACT.md\` and inspect the required workspace files.
2. Ensure \`docs/draft.md\` contains a current plan grounded in this run.
3. Convert the draft into an executable plan in \`docs/plan.md\` before editing implementation files.
4. Implement at least one concrete candidate inside the workspace.
5. Run the validation command from \`TASK_CONTRACT.md\` after each meaningful candidate.
6. If \`FIB_DATASET_PATH\` is set, run the evaluation command for the best validated candidate. If it is not set, stop after validation and record evaluation as blocked by missing dataset.
7. Save command outputs or concise summaries under \`runs/\`.
8. Update \`candidates.jsonl\`, \`benchmark.csv\`, and \`outputs/execution-summary.md\` with the result of the candidate you executed.
9. Keep the final change scoped to the task contract.

Stop when one of the following is true:

- one candidate has been implemented and validated, and evaluation is either completed or explicitly blocked by missing dataset
- you hit a concrete blocker that prevents safe progress
- you need human input to continue

EOF

    if [[ "${KERNEL_OPTIMIZE}" == "1" ]]; then
        cat <<EOF

For this run, stop after either:

- bootstrap completed and ${MAX_DEPTH} optimization candidate(s) were attempted,
- an optimization candidate clearly outperformed the prior baseline and the new
  evidence was recorded,
- or a concrete blocker prevented safe progress.
EOF
    fi

    cat <<EOF

Do not edit files outside the workspace root.
EOF
}

FINAL_PROMPT="$(mktemp)"
cat "${PROMPT_FILE}" > "${FINAL_PROMPT}"
build_workspace_inspection_block >> "${FINAL_PROMPT}"
cat >> "${FINAL_PROMPT}" <<EOF

## OpenShell Run Contract

- Workspace root: ${WORKSPACE}
- Task contract file: ${WORKSPACE}/TASK_CONTRACT.md
- Output draft file: ${WORKSPACE}/docs/draft.md
- Output last message file: ${WORKSPACE}/outputs/last-message.md
EOF

if [[ "${KERNEL_OPTIMIZE}" == "1" ]]; then
    if workspace_has_validated_baseline "${WORKSPACE}/candidates.jsonl" "${WORKSPACE}/runs/run_local_results.json"; then
        KERNEL_OPTIMIZE_STATE="optimize"
    else
        KERNEL_OPTIMIZE_STATE="bootstrap"
    fi
fi

build_execution_mode_block >> "${FINAL_PROMPT}"

cd "${WORKSPACE}"

CODEX_EXEC_ARGS=(
    exec
    --skip-git-repo-check
    --sandbox danger-full-access
    --ephemeral
    --output-last-message "${WORKSPACE}/outputs/last-message.md"
    -c "sqlite_home=\"${CODEX_SQLITE_HOME}\""
    -c "log_dir=\"${KDA_CODEX_LOG_DIR}\""
)

if "${CODEX_BIN}" exec --help 2>/dev/null | grep -q -- "--ignore-user-config"; then
    CODEX_EXEC_ARGS+=(--ignore-user-config)
fi
if "${CODEX_BIN}" exec --help 2>/dev/null | grep -q -- "--ignore-rules"; then
    CODEX_EXEC_ARGS+=(--ignore-rules)
fi

"${CODEX_BIN}" "${CODEX_EXEC_ARGS[@]}" \
    -c "model=\"${MODEL}\"" \
    -c "model_reasoning_effort=\"${REASONING}\"" \
    "$(cat "${FINAL_PROMPT}")"

normalize_benchmark_artifacts() {
    local runs_dir="$1"
    local canonical_log="${runs_dir}/run_local.txt"
    local canonical_json="${runs_dir}/run_local_results.json"
    local candidate

    if [[ ! -f "${canonical_log}" ]]; then
        for candidate in \
            "${runs_dir}/reduced-benchmark.log" \
            "${runs_dir}/benchmark.log"; do
            if [[ -f "${candidate}" ]]; then
                cp "${candidate}" "${canonical_log}"
                break
            fi
        done
    fi

    if [[ ! -f "${canonical_json}" ]]; then
        for candidate in \
            "${runs_dir}/reduced-benchmark-results.json" \
            "${runs_dir}/benchmark-results.json"; do
            if [[ -f "${candidate}" ]]; then
                cp "${candidate}" "${canonical_json}"
                break
            fi
        done
    fi
}

normalize_benchmark_artifacts "${WORKSPACE}/runs"

if [[ ! -s "${WORKSPACE}/docs/draft.md" ]]; then
    echo "Codex exited without producing docs/draft.md" >&2
    exit 1
fi

if [[ "${MODE}" == "execute" && ! -s "${WORKSPACE}/outputs/execution-summary.md" ]]; then
    echo "Codex exited without producing outputs/execution-summary.md" >&2
    exit 1
fi

if [[ "${MODE}" == "execute" ]]; then
    POST_EXECUTION_SUMMARY_HASH="$(cksum "${WORKSPACE}/outputs/execution-summary.md" | awk '{print $1 ":" $2}')"
    if [[ "${PRE_EXECUTION_SUMMARY_HASH}" == "${POST_EXECUTION_SUMMARY_HASH}" ]]; then
        echo "Codex exited without updating outputs/execution-summary.md" >&2
        exit 1
    fi
fi
