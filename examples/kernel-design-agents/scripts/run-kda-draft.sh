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
BRANCH="${KDA_BRANCH:-1}"
BASELINE_STATE="unknown"
SOURCE_STATE="unknown"
ACTIVE_DEFINITION=""
ACTIVE_SOURCE_DIR=""
RUN_PHASE="default"
TARGET_STEP=""
TARGET_BRANCH=""
TARGET_PARENT_STEP=""
TARGET_PARENT_BRANCH=""
TARGET_SEED_STEP=""
TARGET_SEED_BRANCH=""
ATTEMPT_WORKSPACE=""
APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT_SUMMARY_REL="outputs/execution-summary.agent.md"
MACHINE_SUMMARY_REL="outputs/execution-summary.md"

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

case "${MAX_DEPTH}" in
    ''|*[!0-9]*|0)
        echo "Unsupported max depth: ${MAX_DEPTH}. Expected a positive integer." >&2
        exit 2
        ;;
esac

case "${BRANCH}" in
    ''|*[!0-9]*|0)
        echo "Unsupported branch count: ${BRANCH}. Expected a positive integer." >&2
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
touch "${WORKSPACE}/outputs/last-message.md"
if [[ "${MODE}" == "execute" ]]; then
    touch "${WORKSPACE}/candidates.jsonl"
    if [[ ! -f "${WORKSPACE}/benchmark.csv" ]]; then
        printf 'candidate,metric,unit,status,notes\n' > "${WORKSPACE}/benchmark.csv"
    fi
    if [[ ! -f "${WORKSPACE}/${AGENT_SUMMARY_REL}" ]]; then
        cat > "${WORKSPACE}/${AGENT_SUMMARY_REL}" <<'EOF'
# Agent Execution Summary

The agent updates this file after implementation, validation, and optional evaluation.
EOF
    fi
    if [[ ! -f "${WORKSPACE}/${MACHINE_SUMMARY_REL}" ]]; then
        cat > "${WORKSPACE}/${MACHINE_SUMMARY_REL}" <<'EOF'
# Execution Summary

The orchestration layer generates this file from immutable step/branch records.
EOF
    fi
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
    local workspace_root="$1"

    python3 - "${workspace_root}" <<'PY'
import json
import pathlib
import re
import sys

workspace = pathlib.Path(sys.argv[1])
current_path = workspace / "current.json"

def canonical_step_name(step_name: str) -> str:
    text = (step_name or "").strip()
    if text == "baseline":
        return "baseline"
    match = re.fullmatch(r"(?:d|opt-)(\d+)", text)
    if match:
        return f"d{int(match.group(1))}"
    return text

def canonical_branch_name(branch_name: str) -> str:
    text = (branch_name or "").strip()
    match = re.fullmatch(r"(?:b|run)(\d+)", text)
    if match:
        return f"b{int(match.group(1))}"
    return text

def is_passing(record: dict) -> bool:
    status = str(record.get("status", "")).strip().lower()
    correctness = str(record.get("correctness", "")).strip().lower()
    return correctness == "passed" or status in {"validated", "passed", "promoted"}

def step_depth(step_name: str) -> int:
    step_name = canonical_step_name(step_name)
    if step_name == "baseline":
        return 0
    match = re.fullmatch(r"d(\d+)", step_name)
    if match:
        return int(match.group(1))
    return 10**9

def branch_number(branch_name: str) -> int:
    branch_name = canonical_branch_name(branch_name)
    match = re.fullmatch(r"b(\d+)", branch_name)
    if match:
        return int(match.group(1))
    return 10**9

current = None
baseline_dir = workspace / "baseline"
if baseline_dir.is_dir():
    for branch_dir in sorted(baseline_dir.iterdir(), key=lambda path: branch_number(path.name)):
        if not branch_dir.is_dir():
            continue
        if not re.fullmatch(r"(?:b|run)\d+", branch_dir.name):
            continue
        record_path = branch_dir / "record.json"
        if not record_path.exists():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        if is_passing(record):
            current = record

if isinstance(current, dict):
    raise SystemExit(0)

raise SystemExit(1)
PY
    return $?
}

workspace_has_source_candidate() {
    local workspace_root="$1"

    python3 - "${workspace_root}" <<'PY'
import json
import pathlib
import re
import sys

workspace = pathlib.Path(sys.argv[1])
current_path = workspace / "current.json"

def canonical_step_name(step_name: str) -> str:
    text = (step_name or "").strip()
    if text == "baseline":
        return "baseline"
    match = re.fullmatch(r"(?:d|opt-)(\d+)", text)
    if match:
        return f"d{int(match.group(1))}"
    return text

def canonical_branch_name(branch_name: str) -> str:
    text = (branch_name or "").strip()
    match = re.fullmatch(r"(?:b|run)(\d+)", text)
    if match:
        return f"b{int(match.group(1))}"
    return text

def is_passing(record: dict) -> bool:
    status = str(record.get("status", "")).strip().lower()
    correctness = str(record.get("correctness", "")).strip().lower()
    return correctness == "passed" or status in {"validated", "passed", "promoted"}

def step_depth(step_name: str) -> int:
    step_name = canonical_step_name(step_name)
    if step_name == "baseline":
        return 0
    match = re.fullmatch(r"d(\d+)", step_name)
    if match:
        return int(match.group(1))
    return 10**9

def branch_number(branch_name: str) -> int:
    branch_name = canonical_branch_name(branch_name)
    match = re.fullmatch(r"b(\d+)", branch_name)
    if match:
        return int(match.group(1))
    return 10**9

def record_sort_key(record: dict):
    return (
        step_depth(record.get("step") or ""),
        branch_number(record.get("branch") or record.get("run") or ""),
    )

def latest_validated_baseline(records: list[dict]):
    baseline = None
    for record in sorted(records, key=record_sort_key):
        if canonical_step_name(record.get("step")) == "baseline" and is_passing(record):
            baseline = record
    return baseline

def record_key(record: dict):
    return (
        canonical_step_name(record.get("step") or ""),
        canonical_branch_name(record.get("branch") or record.get("run") or ""),
    )

def is_descendant_of(record: dict, ancestor_key: tuple[str, str], record_map: dict[tuple[str, str], dict]) -> bool:
    seen = set()
    current = record
    while isinstance(current, dict):
        key = record_key(current)
        if key == ancestor_key:
            return True
        parent_step = canonical_step_name(current.get("parent_step") or "")
        parent_branch = canonical_branch_name(current.get("parent_branch") or current.get("parent_run") or "")
        if not parent_step or not parent_branch:
            return False
        parent_key = (parent_step, parent_branch)
        if parent_key in seen:
            return False
        seen.add(parent_key)
        current = record_map.get(parent_key)
    return False

records = []
for step_dir in sorted(workspace.iterdir(), key=lambda path: (step_depth(path.name), path.name)):
    if not step_dir.is_dir():
        continue
    if step_dir.name != "baseline" and not re.fullmatch(r"(?:d|opt-)\d+", step_dir.name):
        continue
    step_name = canonical_step_name(step_dir.name)
    for branch_dir in sorted(step_dir.iterdir(), key=lambda path: branch_number(path.name)):
        if not branch_dir.is_dir():
            continue
        if not re.fullmatch(r"(?:b|run)\d+", branch_dir.name):
            continue
        record_path = branch_dir / "record.json"
        if not record_path.exists():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        record["step"] = step_name
        record["branch"] = canonical_branch_name(record.get("branch") or record.get("run") or branch_dir.name)
        if isinstance(record.get("parent_step"), str):
            record["parent_step"] = canonical_step_name(record["parent_step"])
        if isinstance(record.get("parent_branch"), str):
            record["parent_branch"] = canonical_branch_name(record["parent_branch"])
        if isinstance(record.get("parent_run"), str) and not record.get("parent_branch"):
            record["parent_branch"] = canonical_branch_name(record["parent_run"])
        records.append(record)

baseline = latest_validated_baseline(records)
if not isinstance(baseline, dict):
    raise SystemExit(1)

baseline_key = record_key(baseline)
record_map = {record_key(record): record for record in records}
current = baseline
for record in sorted(records, key=record_sort_key):
    if str(record.get("status", "")).strip().lower() != "promoted":
        continue
    if str(record.get("correctness", "")).strip().lower() != "passed":
        continue
    if is_descendant_of(record, baseline_key, record_map):
        current = record

if isinstance(current, dict) and is_passing(current):
    raise SystemExit(0)

raise SystemExit(1)
PY
    return $?
}

read_active_definition() {
    local config_path="$1"

    if [[ ! -f "${config_path}" ]]; then
        return 0
    fi

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
if isinstance(definition, str):
    print(definition.strip())
PY
}

read_active_source_dir() {
    local config_path="$1"

    if [[ ! -f "${config_path}" ]]; then
        return 0
    fi

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

build = data.get("build", {})
source_dir = build.get("source_dir")
language = build.get("language", "")

if isinstance(source_dir, str) and source_dir.strip():
    print(source_dir.strip())
elif isinstance(language, str) and language.strip():
    print(language.strip())
PY
}

step_depth_value() {
    local step_name="$1"
    if [[ "${step_name}" == "baseline" ]]; then
        printf '0\n'
        return 0
    fi
    if [[ "${step_name}" =~ ^d([0-9]+)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "${step_name}" =~ ^opt-([0-9]+)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    printf '0\n'
}

branch_number_value() {
    local branch_name="$1"
    if [[ "${branch_name}" =~ ^b([0-9]+)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "${branch_name}" =~ ^run([0-9]+)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    printf '0\n'
}

current_record_field() {
    local field_name="$1"
    python3 - "${WORKSPACE}" "${field_name}" <<'PY'
import json
import pathlib
import re
import sys

workspace = pathlib.Path(sys.argv[1])
field_name = sys.argv[2]
current_path = workspace / "current.json"

def canonical_step_name(step_name: str) -> str:
    text = (step_name or "").strip()
    if text == "baseline":
        return "baseline"
    match = re.fullmatch(r"(?:d|opt-)(\d+)", text)
    if match:
        return f"d{int(match.group(1))}"
    return text

def canonical_branch_name(branch_name: str) -> str:
    text = (branch_name or "").strip()
    match = re.fullmatch(r"(?:b|run)(\d+)", text)
    if match:
        return f"b{int(match.group(1))}"
    return text

def step_depth(step_name: str) -> int:
    step_name = canonical_step_name(step_name)
    if step_name == "baseline":
        return 0
    match = re.fullmatch(r"d(\d+)", step_name or "")
    if match:
        return int(match.group(1))
    return 10**9

def branch_number(branch_name: str) -> int:
    branch_name = canonical_branch_name(branch_name)
    match = re.fullmatch(r"b(\d+)", branch_name or "")
    if match:
        return int(match.group(1))
    return 10**9

def record_sort_key(record: dict):
    return (
        step_depth(record.get("step") or ""),
        branch_number(record.get("branch") or record.get("run") or ""),
    )

def is_passing(record: dict) -> bool:
    status = str(record.get("status", "")).strip().lower()
    correctness = str(record.get("correctness", "")).strip().lower()
    return correctness == "passed" or status in {"validated", "passed", "promoted"}

def latest_validated_baseline(records: list[dict]):
    baseline = None
    for record in sorted(records, key=record_sort_key):
        if canonical_step_name(record.get("step")) == "baseline" and is_passing(record):
            baseline = record
    return baseline

def record_key(record: dict):
    return (
        canonical_step_name(record.get("step") or ""),
        canonical_branch_name(record.get("branch") or record.get("run") or ""),
    )

def is_descendant_of(record: dict, ancestor_key: tuple[str, str], record_map: dict[tuple[str, str], dict]) -> bool:
    seen = set()
    current = record
    while isinstance(current, dict):
        key = record_key(current)
        if key == ancestor_key:
            return True
        parent_step = canonical_step_name(current.get("parent_step") or "")
        parent_branch = canonical_branch_name(current.get("parent_branch") or current.get("parent_run") or "")
        if not parent_step or not parent_branch:
            return False
        parent_key = (parent_step, parent_branch)
        if parent_key in seen:
            return False
        seen.add(parent_key)
        current = record_map.get(parent_key)
    return False

def current_from_records():
    records = []
    for step_dir in sorted(workspace.iterdir(), key=lambda path: (step_depth(path.name), path.name)):
        if not step_dir.is_dir():
            continue
        if step_dir.name != "baseline" and not re.fullmatch(r"(?:d|opt-)\d+", step_dir.name):
            continue
        step_name = canonical_step_name(step_dir.name)
        for branch_dir in sorted(step_dir.iterdir(), key=lambda path: branch_number(path.name)):
            if not branch_dir.is_dir():
                continue
            if not re.fullmatch(r"(?:b|run)\d+", branch_dir.name):
                continue
            record_path = branch_dir / "record.json"
            if not record_path.exists():
                continue
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            record["step"] = step_name
            record["branch"] = canonical_branch_name(record.get("branch") or record.get("run") or branch_dir.name)
            if isinstance(record.get("parent_step"), str):
                record["parent_step"] = canonical_step_name(record["parent_step"])
            if isinstance(record.get("parent_branch"), str):
                record["parent_branch"] = canonical_branch_name(record["parent_branch"])
            if isinstance(record.get("parent_run"), str) and not record.get("parent_branch"):
                record["parent_branch"] = canonical_branch_name(record["parent_run"])
            records.append(record)

    baseline = latest_validated_baseline(records)
    if not isinstance(baseline, dict):
        return None

    baseline_key = record_key(baseline)
    record_map = {record_key(record): record for record in records}
    current = baseline
    for record in sorted(records, key=record_sort_key):
        if str(record.get("status", "")).strip().lower() != "promoted":
            continue
        if str(record.get("correctness", "")).strip().lower() != "passed":
            continue
        if is_descendant_of(record, baseline_key, record_map):
            current = record
    return current

payload = None
if current_path.exists():
    try:
        payload = json.loads(current_path.read_text(encoding="utf-8"))
    except Exception:
        payload = None

value = None
if field_name in {"source_step", "source_branch"}:
    current = current_from_records()
    if isinstance(current, dict):
        if field_name == "source_step":
            value = current.get("step")
        elif field_name == "source_branch":
            value = current.get("branch") or current.get("run")

if value is None and isinstance(payload, dict):
    value = payload.get(field_name)
    fallbacks = {
        "source_step": ["current_step"],
        "source_branch": ["current_branch", "current_run"],
    }
    if value is None:
        for key in fallbacks.get(field_name, []):
            value = payload.get(key)
            if value is not None:
                break

if isinstance(value, str):
    print(value.strip())
elif isinstance(value, (int, float)):
    print(value)
PY
}

latest_validated_baseline_field() {
    local field_name="$1"
    python3 - "${WORKSPACE}" "${field_name}" <<'PY'
import json
import pathlib
import re
import sys

workspace = pathlib.Path(sys.argv[1])
field_name = sys.argv[2]

def branch_number(name: str) -> int:
    match = re.fullmatch(r"(?:b|run)(\d+)", name or "")
    if match:
        return int(match.group(1))
    return -1

baseline_dir = workspace / "baseline"
best = None
if baseline_dir.is_dir():
    for branch_dir in sorted(baseline_dir.iterdir(), key=lambda path: branch_number(path.name)):
        if not branch_dir.is_dir():
            continue
        record_path = branch_dir / "record.json"
        if not record_path.exists():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        status = str(record.get("status", "")).strip().lower()
        correctness = str(record.get("correctness", "")).strip().lower()
        if correctness == "passed" or status in {"validated", "passed"}:
            best = record

if not isinstance(best, dict):
    raise SystemExit(0)

value = best.get(field_name)
if isinstance(value, str):
    print(value.strip())
elif isinstance(value, (int, float)):
    print(value)
PY
}

record_status_for_attempt() {
    local step_name="$1"
    local branch_name="$2"
    python3 - "${WORKSPACE}" "${step_name}" "${branch_name}" <<'PY'
import json
import pathlib
import sys

workspace = pathlib.Path(sys.argv[1])
step_name = sys.argv[2]
branch_name = sys.argv[3]

candidates = [
    workspace / step_name / branch_name / "record.json",
]

if step_name.startswith("d"):
    legacy_step = f"opt-{step_name[1:]}"
    candidates.append(workspace / legacy_step / branch_name / "record.json")
    if branch_name.startswith("b"):
        candidates.append(workspace / legacy_step / f"run{branch_name[1:]}" / "record.json")
elif step_name == "baseline" and branch_name.startswith("b"):
    candidates.append(workspace / "baseline" / f"run{branch_name[1:]}" / "record.json")

for path in candidates:
    if not path.exists():
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if isinstance(payload, dict):
        status = payload.get("status")
        if isinstance(status, str):
            print(status.strip())
            raise SystemExit(0)
raise SystemExit(0)
PY
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
    local workspace_dir="${ATTEMPT_WORKSPACE:-${WORKSPACE}}"
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
        "docs/task-context.md"
        "docs/draft.md"
        "docs/plan.md"
    )

    printf '\n## Required Workspace Inspection\n\n'
    printf 'Before drafting, read the following files if they exist in the workspace. Use them to justify baseline claims, the active implementation path, and the validation/evaluation commands.\n\n'

    for candidate in "${candidates[@]}"; do
        if [[ -f "${workspace_dir}/${candidate}" ]]; then
            printf -- '- `%s`\n' "${candidate}"
        fi
    done

    printf '\nDo not claim facts about any of those files unless you actually read them in this run.\n'
}

build_execution_mode_block() {
    local workspace_dir="${ATTEMPT_WORKSPACE:-${WORKSPACE}}"
    local results_root="${WORKSPACE}"

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

- Attempt workspace: ${workspace_dir}
- Results root: ${results_root}
- FIB_DATASET_PATH: ${FIB_DATASET_PATH:-<unset>}
- Draft file: ${workspace_dir}/docs/draft.md
- Plan file: ${workspace_dir}/docs/plan.md
- Agent summary file: ${workspace_dir}/${AGENT_SUMMARY_REL}
- Machine summary file: ${results_root}/${MACHINE_SUMMARY_REL}
- Candidate export: ${results_root}/candidates.jsonl
- Benchmark table: ${results_root}/benchmark.csv
- Run artifacts directory: ${workspace_dir}/runs
- Canonical benchmark log path: ${workspace_dir}/runs/run_local.txt
- Canonical benchmark results path: ${workspace_dir}/runs/run_local_results.json
- Target step: ${TARGET_STEP:-<unset>}
- Target branch: ${TARGET_BRANCH:-<unset>}

Submission rules:

- Treat local dataset definitions, workloads, and baseline solutions as semantics and benchmarking references only.
- Do not call runtime APIs from specialized external kernel libraries such as \`flashinfer\` or \`deep_gemm\` unless \`TASK_CONTRACT.md\` explicitly allows it.
- Keep the candidate self-contained under \`solution/\` and aligned with the active \`config.toml\` build path.
- If the active build is CUDA, keep the implementation surface aligned with the configured \`binding\` mode instead of mixing TVM FFI and Torch-extension patterns.
- If the active build is CUDA with \`binding=torch\`, keep \`config.toml\` on a CUDA entry point such as \`kernel.cu::kernel\`. Do not switch the active benchmark path to \`binding.py::...\`.
- Use staged validation for expensive kernels: pack first, then any applicable direct compile smoke, then the reduced benchmark subset configured in the environment, and only then attempt a broader sweep if the candidate still looks viable.
- If \`scripts/check_cuda_extension.py\` exists and the active build is CUDA with \`binding=torch\`, use it for the compile-smoke stage before running \`scripts/run_local.py\`, and pass the artifact path explicitly:
  \`python scripts/check_cuda_extension.py --log-file runs/check_cuda_extension.txt\`
- When you run \`scripts/run_local.py\`, pass the artifact paths explicitly:
  \`python scripts/run_local.py --log-file runs/run_local.txt --results-json runs/run_local_results.json\`
- Do not rely on environment-variable defaults for compile-smoke or benchmark artifact paths.
- Preserve the benchmark artifacts at \`runs/run_local.txt\` and \`runs/run_local_results.json\`. Do not invent alternate filenames.
- Respect the configured workload scope. Use smoke workloads to bootstrap correctness quickly, but do not claim broad results beyond the workloads actually evaluated in this run.
- Do not count edits to \`scripts/pack_solution.py\`, \`scripts/run_local.py\`, or docs-only files as kernel progress unless the contract explicitly asks for tooling work.
- If the candidate returns tensors instead of writing to preallocated outputs, align \`destination_passing_style\` with the actual callable signature.
- Keep the editable implementation under the active \`solution/<source_dir>/...\` tree only. Do not invent your own durable snapshot layout under \`solution/\`.
- The orchestration layer records immutable step/run evidence after the phase ends and generates \`candidates.jsonl\`, \`benchmark.csv\`, and the machine summary. Your job is to keep \`${AGENT_SUMMARY_REL}\` semantically accurate for the candidate you just evaluated.

EOF

    if [[ "${KERNEL_OPTIMIZE}" != "1" ]]; then
        cat <<EOF

## Baseline Smoke Focus

This run is a single baseline smoke or baseline revalidation pass. It is not a kernel optimization run.

- The target for this run is always \`${TARGET_STEP:-baseline}/${TARGET_BRANCH:-b1}\`.
- If a validated baseline already exists in workspace results, start from that baseline as-is and revalidate it first.
- If no validated baseline exists yet, bootstrap one correct self-contained baseline for the active definition.
- Do not spend this run on latency optimization, multi-candidate exploration, or depth/branch search.
- Only repair correctness/build/runtime issues when the untouched baseline fails validation.
- Use \`docs/task-context.md\` as the primary source for definition semantics, selected workload shape, and baseline-reference context when it exists.
- For CUDA with \`binding=torch\`, prefer a straightforward self-contained Torch extension under \`solution/cuda/kernel.cu\` that mirrors the local reference semantics before attempting a lower-level custom kernel.
- Avoid spending the run on FlashInfer-Bench internal API spelunking, alternate bindings, or orchestration edits unless a concrete blocker forces it.
EOF
    fi

    if [[ "${KERNEL_OPTIMIZE}" == "1" ]]; then
        if [[ "${RUN_PHASE}" == "bootstrap-baseline" ]]; then
            cat <<EOF

## Kernel Optimization Bootstrap

This run is the bootstrap-baseline phase for kernel optimization.

- Before any implementation edits, validate the current scaffolded source exactly as it exists in the workspace.
- Record that source as the baseline for this attempt, targeting \`${TARGET_STEP}/${TARGET_BRANCH}\`.
- Do not make latency-oriented implementation edits unless the untouched scaffolded source fails validation.
- If the untouched scaffolded source fails validation, repair baseline correctness first. Describe the failed untouched baseline in \`${AGENT_SUMMARY_REL}\`, but emit only one final candidate record for this attempt.
- Stop this phase as soon as one validated baseline exists in the workspace artifacts.
- Do not spend this phase on optimization follow-ups, alternate bindings, scaffold churn, or prompt-only edits.
EOF
        elif [[ "${KERNEL_OPTIMIZE_STATE}" == "bootstrap" ]]; then
            cat <<EOF

## Kernel Optimization Focus

This run is explicitly for kernel optimization, but the workspace does not yet
have a validated baseline.

- First bootstrap one correct baseline and validate it on the active
  workload subset.
- After that baseline is validated, continue with one optimization attempt for
  the orchestrator-selected target \`${TARGET_STEP}/${TARGET_BRANCH}\`.
- Treat the freshly validated baseline as the source candidate for this
  single optimization attempt.
- Prioritize edits under \`solution/\`, especially the active CUDA or Triton
  implementation files.
- Avoid spending the run on scaffold churn, config rewrites, prompt-only edits,
  or alternate binding experiments unless a concrete blocker forces it.
- Preserve the active build path, callable signature, and validation flow.
- In this phase, "optimize" means improve on the current source candidate for
  the active workload subset while preserving correctness.
EOF
        else
            cat <<EOF

## Kernel Optimization Focus

This run is explicitly for kernel optimization, not baseline bring-up.

- Start from the current source candidate and improve on it for the active workload subset.
- This attempt targets \`${TARGET_STEP}/${TARGET_BRANCH}\`.
- Source candidate: \`${TARGET_PARENT_STEP:-baseline}/${TARGET_PARENT_BRANCH:-b1}\`.
- Prioritize edits under \`solution/\`, especially the active CUDA or Triton implementation files.
- Avoid spending the run on scaffold churn, config rewrites, prompt-only edits, or alternate binding experiments unless a concrete blocker forces it.
- Preserve the active build path, callable signature, and validation flow.
- A slower-or-equal correct candidate should be recorded as a tried candidate for this branch, but it does not count as a successful optimization over baseline.
- Use existing step/branch records, \`runs/run_local_results.json\`, and \`benchmark.csv\` as the before-state when available, then record the new candidate against that source candidate.
EOF

            if [[ "${ACTIVE_DEFINITION}" == "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048" ]]; then
                cat <<EOF

### MoE Optimization Guardrails

- The current validated MoE baseline already has correct routing and dequantization semantics. Treat those semantics as fixed unless you can prove an equivalent transformation locally in this run.
- Do not change the validated routing contract:
  \`s = sigmoid(logits)\`, then \`s_with_bias = s + bias\`, then group top-2 sum, then kept-group global top-k, then normalize combination weights from \`s\` without bias.
- Do not rewrite the FP8 block-scale dequantization layout using a different broadcast or reshape scheme unless you validate the transformed layout against the current source candidate on the selected workload first.
- Prefer safer optimization directions:
  eliminating full \`gemm2\` expert materialization in favor of selected-expert-only dequantization,
  direct accumulation into the output tensor,
  reducing temporary tensor/vector staging,
  hoisting repeated conversions out of inner loops,
  and dequantizing only the selected local experts while preserving the exact existing dequantization semantics.
- Avoid spending an optimization attempt on routing-mask or gather/scatter cleanup alone unless it also removes a larger tensor materialization from the hot path.
- If a candidate changes numerical results on the selected workload, reject it and revert to the current source candidate rather than carrying the optimization forward.
EOF
            fi
        fi
    fi

    cat <<EOF

You must:

1. Read \`TASK_CONTRACT.md\` and inspect the required workspace files.
2. Ensure \`docs/draft.md\` contains a current plan grounded in this run.
3. Convert the draft into an executable plan in \`docs/plan.md\` before editing implementation files.
4. Implement at least one concrete candidate inside the workspace.
5. Run the validation command from \`TASK_CONTRACT.md\` after each meaningful candidate.
6. If a CUDA compile-smoke step is applicable, run it with \`--log-file runs/check_cuda_extension.txt\` before local benchmarking.
7. If \`FIB_DATASET_PATH\` is set, run the evaluation command for the best validated candidate, but ensure the command passes \`--log-file runs/run_local.txt --results-json runs/run_local_results.json\`. If it is not set, stop after validation and record evaluation as blocked by missing dataset.
8. Save command outputs or concise summaries under \`runs/\`.
9. Update only \`${AGENT_SUMMARY_REL}\` with the result of the single candidate you executed for \`${TARGET_STEP}/${TARGET_BRANCH}\`.
10. Do not hand-edit \`candidates.jsonl\` or \`benchmark.csv\`; the orchestration layer regenerates them from immutable step/branch records.
11. Keep the final change scoped to the task contract.

Stop when one of the following is true:

- one candidate has been implemented and validated, and evaluation is either completed or explicitly blocked by missing dataset
- you hit a concrete blocker that prevents safe progress
- you need human input to continue

EOF

    if [[ "${KERNEL_OPTIMIZE}" != "1" ]]; then
        cat <<EOF

For this run, stop after either:

- one baseline candidate for \`${TARGET_STEP}/${TARGET_BRANCH}\` was validated or explicitly failed,
- or a concrete blocker prevented safe baseline validation.
EOF
    elif [[ "${RUN_PHASE}" == "bootstrap-baseline" ]]; then
        cat <<EOF

For this phase, stop after either:

- the untouched scaffolded baseline was validated and recorded,
- the untouched scaffolded baseline failed and a repaired baseline was validated and recorded,
- or a concrete blocker prevented baseline validation.
EOF
    elif [[ "${KERNEL_OPTIMIZE}" == "1" ]]; then
        cat <<EOF

For this attempt, stop after either:

- one candidate for \`${TARGET_STEP}/${TARGET_BRANCH}\` was implemented and evaluated,
- or a concrete blocker prevented safe progress.
EOF
    fi

    cat <<EOF

Do not edit files outside the workspace root.
EOF
}

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

seed_agent_summary_stub() {
    local workspace_dir="$1"
    cat > "${workspace_dir}/${AGENT_SUMMARY_REL}" <<'EOF'
# Agent Execution Summary

The agent updates this file after implementation, validation, and optional evaluation.
EOF
}

reset_attempt_artifacts() {
    local workspace_dir="$1"
    local _phase_name="$2"

    mkdir -p "${workspace_dir}/runs" "${workspace_dir}/outputs"
    : > "${workspace_dir}/outputs/last-message.md"
    seed_agent_summary_stub "${workspace_dir}"
}

snapshot_phase_outputs() {
    local workspace_dir="$1"
    local phase_name="$2"
    local summary_path="${workspace_dir}/${AGENT_SUMMARY_REL}"
    local last_message_path="${workspace_dir}/outputs/last-message.md"

    if [[ -f "${summary_path}" ]]; then
        cp "${summary_path}" "${workspace_dir}/outputs/execution-summary.agent.${phase_name}.md"
    fi
    if [[ -f "${last_message_path}" ]]; then
        cp "${last_message_path}" "${workspace_dir}/outputs/last-message.${phase_name}.md"
    fi
}

canonicalize_workspace_results() {
    python3 - "${WORKSPACE}" "${ATTEMPT_WORKSPACE}" "${TARGET_STEP}" "${TARGET_BRANCH}" "${TARGET_PARENT_STEP}" "${TARGET_PARENT_BRANCH}" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
attempt = pathlib.Path(sys.argv[2])
target_step_name = sys.argv[3]
target_branch_name = sys.argv[4]
target_parent_step = sys.argv[5]
target_parent_branch = sys.argv[6]

candidates_path = root / "candidates.jsonl"
benchmark_path = root / "benchmark.csv"
summary_path = root / "outputs" / "execution-summary.md"
current_path = root / "current.json"
config_path = attempt / "config.toml"
task_contract_path = attempt / "TASK_CONTRACT.md"

def normalize_text(value):
    if isinstance(value, str):
        return value.strip()
    return ""

def normalize_status(value):
    return normalize_text(value).lower()

def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def fmt(value, digits=4):
    if is_number(value):
        return f"{value:.{digits}f}"
    return "-"

def read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_json(path: pathlib.Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

def read_toml(path: pathlib.Path):
    if not path.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_source_dir_from_config(path: pathlib.Path):
    data = read_toml(path)
    build = data.get("build", {})
    source_dir = build.get("source_dir")
    language = build.get("language", "")
    if isinstance(source_dir, str) and source_dir.strip():
        return source_dir.strip()
    if isinstance(language, str) and language.strip():
        return language.strip()
    return ""

def read_definition_from_config(path: pathlib.Path):
    data = read_toml(path)
    definition = data.get("solution", {}).get("definition", "")
    if isinstance(definition, str):
        return definition.strip()
    return ""

def canonical_step_name(step_name: str) -> str:
    text = normalize_text(step_name)
    if text == "baseline":
        return "baseline"
    match = re.fullmatch(r"(?:d|opt-)(\d+)", text)
    if match:
        return f"d{int(match.group(1))}"
    return text

def canonical_branch_name(branch_name: str) -> str:
    text = normalize_text(branch_name)
    match = re.fullmatch(r"(?:b|run)(\d+)", text)
    if match:
        return f"b{int(match.group(1))}"
    return text

def step_depth(step_name: str) -> int:
    step_name = canonical_step_name(step_name)
    if step_name == "baseline":
        return 0
    match = re.fullmatch(r"d(\d+)", step_name or "")
    if match:
        return int(match.group(1))
    return 10**9

def branch_number(branch_name: str) -> int:
    branch_name = canonical_branch_name(branch_name)
    match = re.fullmatch(r"b(\d+)", branch_name or "")
    if match:
        return int(match.group(1))
    return 10**9

def record_sort_key(record):
    return (
        step_depth(record.get("step") or record.get("candidate") or ""),
        branch_number(record.get("branch") or record.get("run") or ""),
    )

def build_display_name(record):
    return f"{record['step']}/{record['branch']}"

def parse_results_metrics(path):
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        return {}

    best = {}
    selected_workloads = data.get("selected_workloads")
    selected_uuid = None
    if isinstance(selected_workloads, list) and len(selected_workloads) == 1 and isinstance(selected_workloads[0], str):
        selected_uuid = selected_workloads[0]

    def maybe_take(obj):
        nonlocal best
        if not isinstance(obj, dict):
            return
        if "latency_ms" not in obj and "speedup_factor" not in obj and "status" not in obj:
            return
        candidate = {
            "latency_ms": obj.get("latency_ms"),
            "reference_latency_ms": obj.get("reference_latency_ms"),
            "speedup_factor": obj.get("speedup_factor"),
            "max_abs_error": obj.get("max_abs_error", obj.get("abs_error")),
            "max_rel_error": obj.get("max_rel_error", obj.get("rel_error")),
            "result_status": obj.get("status"),
            "workload_uuid": selected_uuid,
        }
        if not best:
            best = candidate
            return
        candidate_pass = normalize_status(candidate.get("result_status")) == "passed"
        best_pass = normalize_status(best.get("result_status")) == "passed"
        if candidate_pass and not best_pass:
            best = candidate
            return
        if candidate_pass == best_pass and is_number(candidate.get("latency_ms")) and not is_number(best.get("latency_ms")):
            best = candidate

    def walk(node):
        if isinstance(node, dict):
            maybe_take(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return best

def adopt_root_artifact(primary: pathlib.Path, fallback: pathlib.Path, attempt_marker: str):
    if primary.exists():
        return primary
    if not fallback.exists():
        return primary
    try:
        content = fallback.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return primary
    if attempt_marker and attempt_marker in content:
        return fallback
    return primary

def extract_metrics():
    attempt_marker = f"{attempt.as_posix()}/"
    compile_log = adopt_root_artifact(
        attempt / "runs" / "check_cuda_extension.txt",
        root / "runs" / "check_cuda_extension.txt",
        attempt_marker,
    )
    run_local_log = adopt_root_artifact(
        attempt / "runs" / "run_local.txt",
        root / "runs" / "run_local.txt",
        attempt_marker,
    )
    run_local_results = attempt / "runs" / "run_local_results.json"
    if not run_local_results.exists():
        root_results = root / "runs" / "run_local_results.json"
        if root_results.exists() and run_local_log == root / "runs" / "run_local.txt":
            run_local_results = root_results

    metrics = {
        "latency_ms": None,
        "reference_latency_ms": None,
        "speedup_factor": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "result_status": None,
        "workload_uuid": None,
    }

    parsed = parse_results_metrics(run_local_results)
    for key in ("latency_ms", "reference_latency_ms", "speedup_factor", "max_abs_error", "max_rel_error", "result_status", "workload_uuid"):
        if key in parsed:
            metrics[key] = parsed[key]

    if metrics["speedup_factor"] is None and is_number(metrics["latency_ms"]) and is_number(metrics["reference_latency_ms"]) and metrics["latency_ms"] > 0:
        metrics["speedup_factor"] = metrics["reference_latency_ms"] / metrics["latency_ms"]

    status_candidates = []
    if run_local_results.exists():
        status_candidates.append(str(parsed.get("result_status") or ""))
    if run_local_log.exists():
        try:
            status_candidates.append(run_local_log.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    if compile_log.exists():
        try:
            status_candidates.append(compile_log.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass

    lowered = " ".join(normalize_status(value) for value in status_candidates if value)
    if "incorrect_numerical" in lowered or "runtime_error" in lowered or "failed" in lowered:
        correctness = "failed"
    elif "passed" in lowered or normalize_status(metrics.get("result_status")) == "passed":
        correctness = "passed"
    else:
        correctness = "unknown"

    return {
        "compile_log": compile_log if compile_log.exists() else None,
        "run_local_log": run_local_log if run_local_log.exists() else None,
        "run_local_results": run_local_results if run_local_results.exists() else None,
        "metrics": metrics,
        "correctness": correctness,
    }

def load_existing_records():
    records = []
    for step_dir in sorted(root.iterdir(), key=lambda path: (step_depth(path.name), path.name)):
        if not step_dir.is_dir():
            continue
        if step_dir.name != "baseline" and not re.fullmatch(r"(?:d|opt-)\d+", step_dir.name):
            continue
        for branch_dir in sorted(step_dir.iterdir(), key=lambda path: branch_number(path.name)):
            if not branch_dir.is_dir() or not re.fullmatch(r"(?:b|run)\d+", branch_dir.name):
                continue
            record_path = branch_dir / "record.json"
            if not record_path.exists():
                continue
            record = read_json(record_path)
            if not isinstance(record, dict):
                continue
            record["step"] = canonical_step_name(record.get("step") or step_dir.name)
            record["candidate"] = record["step"]
            record["branch"] = canonical_branch_name(record.get("branch") or record.get("run") or branch_dir.name)
            record.pop("run", None)
            if isinstance(record.get("parent_step"), str):
                record["parent_step"] = canonical_step_name(record["parent_step"])
            if isinstance(record.get("parent_run"), str) and not record.get("parent_branch"):
                record["parent_branch"] = canonical_branch_name(record["parent_run"])
            if isinstance(record.get("parent_branch"), str):
                record["parent_branch"] = canonical_branch_name(record["parent_branch"])
            record.pop("parent_run", None)
            if isinstance(record.get("parent"), str):
                record["parent"] = record["parent"].replace("opt-", "d").replace("/run", "/b")
            records.append(record)
    records.sort(key=record_sort_key)
    return records

def resolve_record(records, step_name, branch_name):
    wanted_step = canonical_step_name(step_name)
    wanted_branch = canonical_branch_name(branch_name)
    if not wanted_step or not wanted_branch:
        return None
    for record in records:
        if canonical_step_name(record.get("step")) == wanted_step and canonical_branch_name(record.get("branch")) == wanted_branch:
            return record
    return None

def latest_validated_baseline(records):
    baseline = None
    for record in sorted(records, key=record_sort_key):
        if canonical_step_name(record.get("step")) == "baseline" and record.get("correctness") == "passed":
            baseline = record
    return baseline

def resolve_parent(record, record_map):
    if not isinstance(record, dict):
        return None
    parent_step = canonical_step_name(record.get("parent_step") or "")
    parent_branch = canonical_branch_name(record.get("parent_branch") or record.get("parent_run") or "")
    if not parent_step or not parent_branch:
        return None
    return record_map.get((parent_step, parent_branch))

def baseline_ancestor(record, record_map):
    current = record
    seen = set()
    while isinstance(current, dict):
        if canonical_step_name(current.get("step")) == "baseline":
            return current
        key = record_key(current)
        if key in seen:
            return None
        seen.add(key)
        current = resolve_parent(current, record_map)
    return None

def record_key(record):
    return (
        canonical_step_name(record.get("step") or ""),
        canonical_branch_name(record.get("branch") or record.get("run") or ""),
    )

def is_descendant_of(record, ancestor_key, record_map):
    seen = set()
    current = record
    while isinstance(current, dict):
        key = record_key(current)
        if key == ancestor_key:
            return True
        parent_step = canonical_step_name(current.get("parent_step") or "")
        parent_branch = canonical_branch_name(current.get("parent_branch") or current.get("parent_run") or "")
        if not parent_step or not parent_branch:
            return False
        parent_key = (parent_step, parent_branch)
        if parent_key in seen:
            return False
        seen.add(parent_key)
        current = record_map.get(parent_key)
    return False

def recompute_current(records):
    current = latest_validated_baseline(records)
    if current is None:
        return None
    baseline_key = record_key(current)
    record_map = {record_key(record): record for record in records}
    for record in sorted(records, key=record_sort_key):
        record["source_candidate"] = False
        if (
            str(record.get("status", "")).strip().lower() == "promoted"
            and str(record.get("correctness", "")).strip().lower() == "passed"
            and is_descendant_of(record, baseline_key, record_map)
        ):
            current = record
    if current is not None:
        current["source_candidate"] = True
    return current

def build_notes(metrics):
    parts = []
    if metrics.get("workload_uuid"):
        parts.append(f"workload={metrics['workload_uuid']}")
    if is_number(metrics.get("latency_ms")):
        parts.append(f"latency_ms={metrics['latency_ms']:.6f}")
    if is_number(metrics.get("reference_latency_ms")):
        parts.append(f"reference_latency_ms={metrics['reference_latency_ms']:.6f}")
    if is_number(metrics.get("speedup_factor")):
        parts.append(f"speedup_factor={metrics['speedup_factor']:.6f}")
    if is_number(metrics.get("max_abs_error")):
        parts.append(f"abs_err={metrics['max_abs_error']}")
    if is_number(metrics.get("max_rel_error")):
        parts.append(f"rel_err={metrics['max_rel_error']}")
    return "; ".join(parts)

def write_candidates_index(records):
    with candidates_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=record_sort_key):
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")

def write_benchmark(records):
    lines = ["candidate,metric,unit,status,notes"]
    for record in sorted(records, key=record_sort_key):
        metrics = record.get("metrics", {})
        display_name = build_display_name(record)
        correctness_status = "passed" if record.get("correctness") == "passed" else "failed"
        lines.append(f'{display_name},correctness,workload,{correctness_status},"{build_notes(metrics)}"')
        if record.get("step") == "baseline":
            continue
        source_latency = record.get("source_latency_ms", record.get("baseline_latency_ms"))
        latency = metrics.get("latency_ms")
        notes = []
        if record.get("parent"):
            notes.append(f"parent={record['parent']}")
        if is_number(latency):
            notes.append(f"latency_ms={latency:.6f}")
        if is_number(source_latency):
            notes.append(f"source_latency_ms={source_latency:.6f}")
            if is_number(latency):
                notes.append(f"delta_ms={(latency - source_latency):.6f}")
        lines.append(f'{display_name},optimize,step,{record.get("status", "unknown")},"{"; ".join(notes)}"')
    benchmark_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_summary(records, current_record):
    lines = ["# Execution Summary", ""]
    if current_record is None:
        lines.extend(["## Source Candidate", "- No source candidate yet.", ""])
    else:
        metrics = current_record.get("metrics", {})
        lines.extend([
            "## Source Candidate",
            f"- Step: `{current_record['step']}`",
            f"- Branch: `{current_record['branch']}`",
            f"- Source: `{current_record['solution_snapshot_dir']}`",
            f"- Latency: `{fmt(metrics.get('latency_ms'))} ms`",
            f"- Speedup: `{fmt(metrics.get('speedup_factor'))}x`",
            "",
            "## Step Runs",
        ])
        for record in sorted(records, key=record_sort_key):
            metrics = record.get("metrics", {})
            parts = [
                f"- `{record['step']}/{record['branch']}`",
                f"status `{record.get('status', '-')}`",
                f"correctness `{record.get('correctness', '-')}`",
                f"latency `{fmt(metrics.get('latency_ms'))} ms`",
            ]
            if record.get("parent"):
                parts.append(f"parent `{record['parent']}`")
            source_latency = record.get("source_latency_ms", record.get("baseline_latency_ms"))
            if is_number(source_latency) and is_number(metrics.get("latency_ms")):
                parts.append(f"delta `{metrics['latency_ms'] - source_latency:.4f} ms`")
            if record.get("source_candidate"):
                parts.append("source `true`")
            lines.append("; ".join(parts))
        lines.append("")

    attempt_summaries = []
    for record in sorted(records, key=record_sort_key):
        evidence = record.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        summary_rel = evidence.get("agent_execution_summary")
        if isinstance(summary_rel, str) and summary_rel:
            attempt_summaries.append((record["step"], record["branch"], summary_rel))
    if attempt_summaries:
        lines.append("## Attempt Artifacts")
        for step_name, branch_name, summary_rel in attempt_summaries:
            lines.append(f"- `{step_name}/{branch_name}` summary: `{summary_rel}`")
        lines.append("")

    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

records = load_existing_records()
current_record = recompute_current(records)
record_map = {record_key(record): record for record in records}
requested_step = canonical_step_name(target_step_name)
requested_branch = canonical_branch_name(target_branch_name)
requested_parent = resolve_record(records, target_parent_step, target_parent_branch)
step_name = requested_step or ("baseline" if current_record is None else f"d{step_depth(current_record['step']) + 1}")
branch_name = requested_branch or canonical_branch_name(attempt.name)

if resolve_record(records, step_name, branch_name) is not None:
    raise SystemExit(f"attempt record already exists for {step_name}/{branch_name}")

prepared = extract_metrics()
source_dir_name = read_source_dir_from_config(config_path) or "cuda"
definition_name = read_definition_from_config(config_path)
solution_target = attempt / "solution" / source_dir_name

if step_name == "baseline":
    parent_record = None
    baseline_record = None
    correctness = prepared["correctness"]
    status = "validated" if correctness == "passed" else "failed"
else:
    parent_record = requested_parent or current_record
    if parent_record is None:
        raise SystemExit("optimization attempt requested without a source candidate record")
    baseline_record = baseline_ancestor(parent_record, record_map) or latest_validated_baseline(records)
    correctness = prepared["correctness"]
    latency = prepared["metrics"].get("latency_ms")
    source_latency = parent_record.get("metrics", {}).get("latency_ms")
    improves = correctness == "passed" and is_number(latency) and is_number(source_latency) and latency < source_latency
    if correctness != "passed":
        status = "failed"
    elif improves:
        status = "promoted"
    else:
        status = "rejected"

parent_display = build_display_name(parent_record) if parent_record else None
baseline_display = build_display_name(baseline_record) if baseline_record else None
record = {
    "record_version": 3,
    "candidate": step_name,
    "step": step_name,
    "branch": branch_name,
    "candidate_original": f"{step_name}/{branch_name}",
    "parent": parent_display,
    "parent_step": parent_record.get("step") if parent_record else None,
    "parent_branch": parent_record.get("branch") if parent_record else None,
    "status": status,
    "definition": definition_name,
    "source_dir": str(solution_target.relative_to(root)),
    "solution_snapshot_dir": str(solution_target.relative_to(root)),
    "config_snapshot": str(config_path.relative_to(root)) if config_path.exists() else None,
    "task_contract_snapshot": str(task_contract_path.relative_to(root)) if task_contract_path.exists() else None,
    "artifacts_dir": str(attempt.relative_to(root)),
    "correctness": correctness,
    "improves_source": status == "promoted",
    "improves_baseline": status == "promoted",
    "source_candidate": False,
    "source_compared_to": parent_display,
    "baseline_compared_to": baseline_display,
    "source_latency_ms": parent_record.get("metrics", {}).get("latency_ms") if parent_record else None,
    "baseline_latency_ms": baseline_record.get("metrics", {}).get("latency_ms") if baseline_record else None,
    "metrics": {
        "latency_ms": prepared["metrics"].get("latency_ms"),
        "reference_latency_ms": prepared["metrics"].get("reference_latency_ms"),
        "speedup_factor": prepared["metrics"].get("speedup_factor"),
        "max_abs_error": prepared["metrics"].get("max_abs_error"),
        "max_rel_error": prepared["metrics"].get("max_rel_error"),
        "workload_uuid": prepared["metrics"].get("workload_uuid"),
    },
    "evidence": {
        "evaluation": "PASSED" if correctness == "passed" else normalize_text(prepared["metrics"].get("result_status")) or "FAILED",
        "compile_log": str(prepared["compile_log"].relative_to(root)) if prepared["compile_log"] else None,
        "run_local_log": str(prepared["run_local_log"].relative_to(root)) if prepared["run_local_log"] else None,
        "run_local_results": str(prepared["run_local_results"].relative_to(root)) if prepared["run_local_results"] else None,
        "solution_snapshot_dir": str(solution_target.relative_to(root)),
        "agent_execution_summary": str((attempt / "outputs" / "execution-summary.agent.md").relative_to(root)) if (attempt / "outputs" / "execution-summary.agent.md").exists() else None,
        "agent_last_message": str((attempt / "outputs" / "last-message.md").relative_to(root)) if (attempt / "outputs" / "last-message.md").exists() else None,
    },
}

write_json(attempt / "record.json", record)
records.append(record)
records.sort(key=record_sort_key)
current_record = recompute_current(records)

for item in records:
    write_json(root / item["step"] / item["branch"] / "record.json", item)

write_candidates_index(records)
write_benchmark(records)
write_summary(records, current_record)

if current_record is None:
    if current_path.exists():
        current_path.unlink()
else:
    write_json(current_path, {
        "version": 3,
        "source_step": current_record["step"],
        "source_branch": current_record["branch"],
        "current_step": current_record["step"],
        "current_branch": current_record["branch"],
        "solution_snapshot_dir": current_record["solution_snapshot_dir"],
        "config_snapshot": current_record.get("config_snapshot"),
        "task_contract_snapshot": current_record.get("task_contract_snapshot"),
        "artifacts_dir": current_record["artifacts_dir"],
        "definition": current_record.get("definition"),
        "metrics": current_record.get("metrics", {}),
    })
PY
}

merge_phase_execution_summaries() {
    python3 - "${WORKSPACE}" "${AGENT_SUMMARY_REL}" <<'PY'
import json
import pathlib
import re
import sys

workspace = pathlib.Path(sys.argv[1])
target = workspace / sys.argv[2]

def step_depth(name: str) -> int:
    if name == "baseline":
        return 0
    match = re.fullmatch(r"d(\d+)", name or "")
    if match:
        return int(match.group(1))
    legacy = re.fullmatch(r"opt-(\d+)", name or "")
    if legacy:
        return int(legacy.group(1))
    return 10**9

def branch_number(name: str) -> int:
    match = re.fullmatch(r"(?:b|run)(\d+)", name or "")
    if match:
        return int(match.group(1))
    return 10**9

entries = []
for step_dir in sorted(workspace.iterdir(), key=lambda path: (step_depth(path.name), path.name)):
    if not step_dir.is_dir():
        continue
    if step_dir.name != "baseline" and not re.fullmatch(r"(?:d|opt-)\d+", step_dir.name):
        continue
    for branch_dir in sorted(step_dir.iterdir(), key=lambda path: branch_number(path.name)):
        if not branch_dir.is_dir():
            continue
        if not re.fullmatch(r"(?:b|run)\d+", branch_dir.name):
            continue
        record_path = branch_dir / "record.json"
        summary_path = branch_dir / "outputs" / "execution-summary.agent.md"
        if not record_path.exists() or not summary_path.exists():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            record = {}
        step = record.get("step") or step_dir.name
        branch = record.get("branch") or branch_dir.name
        entries.append((step_depth(step), branch_number(branch), step, branch, summary_path))

if not entries:
    raise SystemExit(0)

lines = ["# Agent Execution Summary", ""]
for _, _, step, branch, path in entries:
    lines.append(f"## {step}/{branch}")
    lines.append("")
    body = path.read_text(encoding="utf-8").strip()
    if body.startswith("# Agent Execution Summary"):
        body = body.split("\n", 1)[1].lstrip() if "\n" in body else ""
    lines.append(body if body else "(empty)")
    lines.append("")

target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY
}

refresh_run_state() {
    if workspace_has_validated_baseline "${WORKSPACE}"; then
        BASELINE_STATE="present"
    else
        BASELINE_STATE="absent"
    fi

    if workspace_has_source_candidate "${WORKSPACE}"; then
        SOURCE_STATE="present"
    else
        SOURCE_STATE="absent"
    fi

    local config_workspace="${ATTEMPT_WORKSPACE:-${WORKSPACE}}"
    ACTIVE_DEFINITION="$(read_active_definition "${config_workspace}/config.toml" || true)"
    ACTIVE_SOURCE_DIR="$(read_active_source_dir "${config_workspace}/config.toml" || true)"
    if [[ -z "${ACTIVE_SOURCE_DIR}" ]]; then
        ACTIVE_SOURCE_DIR="cuda"
    fi

    KERNEL_OPTIMIZE_STATE="disabled"
    if [[ "${KERNEL_OPTIMIZE}" == "1" ]]; then
        if [[ "${SOURCE_STATE}" == "present" ]]; then
            KERNEL_OPTIMIZE_STATE="optimize"
        else
            KERNEL_OPTIMIZE_STATE="bootstrap"
        fi
    fi
}

set_attempt_context() {
    TARGET_STEP="$1"
    TARGET_BRANCH="$2"
    TARGET_PARENT_STEP="$3"
    TARGET_PARENT_BRANCH="$4"
    TARGET_SEED_STEP="${5:-}"
    TARGET_SEED_BRANCH="${6:-}"
}

materialize_attempt_workspace() {
    ATTEMPT_WORKSPACE="${WORKSPACE}/${TARGET_STEP}/${TARGET_BRANCH}"

    python3 - "${WORKSPACE}" "${ATTEMPT_WORKSPACE}" "${TARGET_SEED_STEP}" "${TARGET_SEED_BRANCH}" "${ACTIVE_DEFINITION}" "${ACTIVE_SOURCE_DIR}" "${APP_ROOT}" <<'PY'
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])
attempt = pathlib.Path(sys.argv[2])
seed_step = sys.argv[3].strip()
seed_branch = sys.argv[4].strip()
active_definition = sys.argv[5].strip()
active_source_dir = sys.argv[6].strip() or "cuda"
app_root = pathlib.Path(sys.argv[7])

def copy_path(source: pathlib.Path, target: pathlib.Path):
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

if attempt.exists():
    raise SystemExit(f"attempt workspace already exists: {attempt}")

seed_root = root
if seed_step and seed_branch:
    candidate = root / seed_step / seed_branch
    if candidate.exists():
        seed_root = candidate

attempt.mkdir(parents=True, exist_ok=True)

for name in ("README.md", "FAQ.md", "EVALUATION.md", "TASK_CONTRACT.md"):
    copy_path(root / name, attempt / name)

config_source = seed_root / "config.toml"
if not config_source.exists() and seed_root != root:
    config_source = root / "config.toml"
copy_path(config_source, attempt / "config.toml")

for name in ("scripts", "images"):
    copy_path(root / name, attempt / name)

solution_source = seed_root / "solution"
if not solution_source.exists() and seed_root != root:
    solution_source = root / "solution"
copy_path(solution_source, attempt / "solution")

copy_path(root / "docs" / "task-context.md", attempt / "docs" / "task-context.md")

if not (attempt / "solution").exists() and active_definition:
    template_dir = app_root / "definition-templates" / active_definition / "solution"
    if active_source_dir and (template_dir / active_source_dir).exists():
        copy_path(template_dir, attempt / "solution")

for name in ("docs", "runs", "outputs", "profile"):
    (attempt / name).mkdir(parents=True, exist_ok=True)

(attempt / "outputs" / "last-message.md").touch()
PY
}

next_branch_for_step() {
    local step_name="$1"
    python3 - "${WORKSPACE}" "${step_name}" <<'PY'
import pathlib
import re
import sys

workspace = pathlib.Path(sys.argv[1])
step_name = sys.argv[2]
max_branch = 0
step_dirs = [workspace / step_name]
if step_name.startswith("d"):
    step_dirs.append(workspace / f"opt-{step_name[1:]}")
for step_dir in step_dirs:
    if not step_dir.is_dir():
        continue
    for child in step_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"(?:b|run)(\d+)", child.name)
        if match:
            max_branch = max(max_branch, int(match.group(1)))
print(f"b{max_branch + 1}")
PY
}

assert_attempt_record_exists() {
    local status
    status="$(record_status_for_attempt "${TARGET_STEP}" "${TARGET_BRANCH}" || true)"
    if [[ -z "${status}" ]]; then
        echo "Missing attempt record for ${TARGET_STEP}/${TARGET_BRANCH}" >&2
        exit 1
    fi
    printf '%s\n' "${status}"
}

run_bootstrap_baseline_loop() {
    local attempt_count=0 branch_name

    while (( attempt_count < BRANCH )); do
        branch_name="$(next_branch_for_step "baseline")"
        set_attempt_context "baseline" "${branch_name}" "" "" "" ""
        RUN_PHASE="bootstrap-baseline"
        run_codex_phase
        refresh_run_state
        if [[ "${BASELINE_STATE}" == "present" ]]; then
            return 0
        fi
        attempt_count=$((attempt_count + 1))
    done

    return 1
}

run_optimize_depth_loop() {
    local promoted_levels=0

    while (( promoted_levels < MAX_DEPTH )); do
        local parent_step parent_branch next_depth attempt_count branch_name status promoted_this_depth
        parent_step="$(current_record_field source_step || true)"
        parent_branch="$(current_record_field source_branch || true)"

        if [[ -z "${parent_step}" ]]; then
            echo "Kernel optimize requested without a validated source candidate." >&2
            exit 1
        fi

        next_depth=$(( $(step_depth_value "${parent_step}") + 1 ))
        promoted_this_depth=0
        attempt_count=0

        while (( attempt_count < BRANCH )); do
            branch_name="$(next_branch_for_step "d${next_depth}")"
            set_attempt_context "d${next_depth}" "${branch_name}" "${parent_step}" "${parent_branch}" "${parent_step}" "${parent_branch}"
            RUN_PHASE="optimize"
            run_codex_phase
            status="$(assert_attempt_record_exists)"
            refresh_run_state
            if [[ "${status}" == "promoted" ]]; then
                promoted_levels=$((promoted_levels + 1))
                promoted_this_depth=1
                break
            fi
            attempt_count=$((attempt_count + 1))
        done

        if (( promoted_this_depth == 0 )); then
            return 0
        fi
    done

    return 0
}

run_codex_phase() {
    local phase_summary_hash final_prompt phase_name exec_workspace

    phase_name="${RUN_PHASE}"

    refresh_run_state
    if [[ "${MODE}" == "execute" ]]; then
        materialize_attempt_workspace
    else
        ATTEMPT_WORKSPACE=""
    fi
    exec_workspace="${ATTEMPT_WORKSPACE:-${WORKSPACE}}"
    refresh_run_state
    phase_summary_hash=""
    if [[ "${MODE}" == "execute" ]]; then
        reset_attempt_artifacts "${exec_workspace}" "${phase_name}"
        phase_summary_hash="$(cksum "${exec_workspace}/${AGENT_SUMMARY_REL}" | awk '{print $1 ":" $2}')"
    fi

    final_prompt="$(mktemp)"
    cat "${PROMPT_FILE}" > "${final_prompt}"
    build_workspace_inspection_block >> "${final_prompt}"
    cat >> "${final_prompt}" <<EOF

## OpenShell Run Contract

- Workspace root: ${exec_workspace}
- Results root: ${WORKSPACE}
- Task contract file: ${exec_workspace}/TASK_CONTRACT.md
- Output draft file: ${exec_workspace}/docs/draft.md
- Output last message file: ${exec_workspace}/outputs/last-message.md
EOF

    build_execution_mode_block >> "${final_prompt}"

    cd "${exec_workspace}"

    CODEX_EXEC_ARGS=(
        exec
        --skip-git-repo-check
        --sandbox danger-full-access
        --ephemeral
        --output-last-message "${exec_workspace}/outputs/last-message.md"
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
        "$(cat "${final_prompt}")"

    if [[ ! -s "${exec_workspace}/docs/draft.md" ]]; then
        echo "Codex exited without producing docs/draft.md" >&2
        exit 1
    fi

    if [[ "${MODE}" == "execute" ]]; then
        if [[ ! -s "${exec_workspace}/${AGENT_SUMMARY_REL}" ]]; then
            echo "Codex exited without producing ${AGENT_SUMMARY_REL}" >&2
            exit 1
        fi

        local post_execution_summary_hash
        post_execution_summary_hash="$(cksum "${exec_workspace}/${AGENT_SUMMARY_REL}" | awk '{print $1 ":" $2}')"
        if [[ "${phase_summary_hash}" == "${post_execution_summary_hash}" ]]; then
            echo "Codex exited without updating ${AGENT_SUMMARY_REL}" >&2
            exit 1
        fi

        normalize_benchmark_artifacts "${exec_workspace}/runs"
        snapshot_phase_outputs "${exec_workspace}" "${phase_name}"
        canonicalize_workspace_results
        if [[ -f "${exec_workspace}/outputs/last-message.md" ]]; then
            cp "${exec_workspace}/outputs/last-message.md" "${WORKSPACE}/outputs/last-message.md"
        fi
        merge_phase_execution_summaries
        ATTEMPT_WORKSPACE=""
    fi
}

RUN_PHASE="default"
if [[ "${MODE}" == "execute" && "${KERNEL_OPTIMIZE}" == "1" ]]; then
    refresh_run_state
    if [[ "${SOURCE_STATE}" == "absent" ]]; then
        if ! run_bootstrap_baseline_loop; then
            echo "Bootstrap baseline phase finished without producing a validated baseline." >&2
            exit 1
        fi
        run_optimize_depth_loop
        merge_phase_execution_summaries
        exit 0
    fi
    run_optimize_depth_loop
    merge_phase_execution_summaries
    exit 0
fi

refresh_run_state
if [[ -z "${TARGET_STEP}" ]]; then
    if [[ "${MODE}" == "execute" ]]; then
        baseline_seed_branch=""
        baseline_seed_step=""
        if [[ "${BASELINE_STATE}" == "present" ]]; then
            baseline_seed_branch="$(latest_validated_baseline_field branch || true)"
            if [[ -n "${baseline_seed_branch}" ]]; then
                baseline_seed_step="baseline"
            fi
        fi
        set_attempt_context "baseline" "$(next_branch_for_step "baseline")" "" "" "${baseline_seed_step}" "${baseline_seed_branch}"
    elif [[ "${BASELINE_STATE}" != "present" ]]; then
        set_attempt_context "baseline" "b1" "" "" "" ""
    fi
fi
run_codex_phase
