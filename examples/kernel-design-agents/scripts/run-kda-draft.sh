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

usage() {
    cat <<'EOF'
Usage: /bin/bash /app/scripts/run-kda-draft.sh --workspace <path> [options]

Options:
  --workspace <path>   Uploaded task workspace inside the sandbox.
  --model <name>       Codex model override.
  --reasoning <level>  Codex reasoning effort.
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
        "TASK_CONTRACT.md"
        "config.toml"
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

FINAL_PROMPT="$(mktemp)"
cat "${PROMPT_FILE}" > "${FINAL_PROMPT}"
build_workspace_inspection_block >> "${FINAL_PROMPT}"
cat >> "${FINAL_PROMPT}" <<EOF

## OpenShell Run Contract

- Workspace root: ${WORKSPACE}
- Task contract file: ${WORKSPACE}/TASK_CONTRACT.md
- Output draft file: ${WORKSPACE}/docs/draft.md
- Output last message file: ${WORKSPACE}/outputs/last-message.md

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

if [[ ! -s "${WORKSPACE}/docs/draft.md" ]]; then
    echo "Codex exited without producing docs/draft.md" >&2
    exit 1
fi
