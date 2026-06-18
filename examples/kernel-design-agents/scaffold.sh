#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE=0
TARGET=""
TEMPLATE_DIR="${ROOT}/starter-kit"

usage() {
    cat <<'EOF'
Usage: bash examples/kernel-design-agents/scaffold.sh [options] <workspace-dir>

Creates a local Kernel Design Agents task workspace from the vendored
FlashInfer starter kit and adds the KDA bookkeeping files.

Options:
  --force             Overwrite scaffold-managed files in an existing workspace.
  --template=<path>   Override the starter-kit template source.
  -h, --help          Show this help.
EOF
}

for arg in "$@"; do
    case "${arg}" in
        --force)
            FORCE=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --template=*)
            TEMPLATE_DIR="${arg#--template=}"
            ;;
        *)
            if [[ -n "${TARGET}" ]]; then
                echo "Unexpected argument: ${arg}" >&2
                usage >&2
                exit 2
            fi
            TARGET="${arg}"
            ;;
    esac
done

if [[ -z "${TARGET}" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -d "${TEMPLATE_DIR}" ]]; then
    echo "Starter-kit template directory not found: ${TEMPLATE_DIR}" >&2
    exit 2
fi

mkdir -p "${TARGET}"

if [[ "${FORCE}" != "1" ]]; then
    if [[ -e "${TARGET}/TASK_CONTRACT.md" || -e "${TARGET}/config.toml" || -e "${TARGET}/docs/draft.md" ]]; then
        echo "Workspace already looks initialized: ${TARGET}" >&2
        echo "Use --force to overwrite the scaffolded files." >&2
        exit 2
    fi
fi

if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude '.git' "${TEMPLATE_DIR}/" "${TARGET}/"
else
    (
        cd "${TEMPLATE_DIR}"
        tar --exclude='.git' -cf - .
    ) | (
        cd "${TARGET}"
        tar -xf -
    )
fi

mkdir -p "${TARGET}/docs" "${TARGET}/runs" "${TARGET}/outputs" "${TARGET}/profile"

cat > "${TARGET}/TASK_CONTRACT.md" <<'EOF'
# Task Contract

- Task name: <fill in>
- Objective: <fill in the user-facing goal>
- Correctness requirements: <fill in required behavior, tolerances, or invariants>
- Performance or quality target: <fill in measurable target if any>
- Allowed implementation approaches: <fill in languages, libraries, APIs, or constraints>
- Validation command: <fill in the command that proves correctness>
- Evaluation command: <fill in the command that measures the target, if different>
- Promotion criteria: <fill in what must be true before a candidate is accepted>
EOF

cat > "${TARGET}/docs/draft.md" <<'EOF'
# Draft Plan

The agent writes the first KDA draft here after inspecting the workspace and
reading `TASK_CONTRACT.md`.
EOF

cat > "${TARGET}/docs/plan.md" <<'EOF'
# Executable Plan

Convert `docs/draft.md` into an executable plan before implementation starts.
EOF

cat > "${TARGET}/benchmark.csv" <<'EOF'
candidate,metric,unit,status,notes
EOF

: > "${TARGET}/candidates.jsonl"

cp "${ROOT}/prompts/basic-flow.md" "${TARGET}/docs/kda-basic-flow.md"

echo "Scaffolded KDA workspace from starter kit: ${TARGET}"
