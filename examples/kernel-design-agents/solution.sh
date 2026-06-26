#!/usr/bin/env bash
set -euo pipefail

ROOT_ARG="${1:-examples/kernel-design-agents/results}"

python3 - "$ROOT_ARG" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve()
cwd = pathlib.Path.cwd().resolve()
if not root.exists():
    raise SystemExit(f"results path does not exist: {root}")

def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except Exception:
        return str(path)

def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def fmt(value, digits=4):
    if is_number(value):
        return f"{value:.{digits}f}"
    return "-"

def step_depth(step_name: str) -> int:
    if step_name == "baseline":
        return 0
    match = re.fullmatch(r"(?:d|opt-)(\d+)", step_name or "")
    if match:
        return int(match.group(1))
    return 10**9

def branch_number(branch_name: str) -> int:
    match = re.fullmatch(r"(?:b|run)(\d+)", branch_name or "")
    if match:
        return int(match.group(1))
    return 10**9

def iter_result_dirs(base: pathlib.Path):
    if (base / "current.json").exists() or (base / "candidates.jsonl").exists():
        yield base
        return
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if (child / "current.json").exists() or (child / "candidates.jsonl").exists():
            yield child

def load_step_records(result_dir: pathlib.Path):
    records = []
    for step_dir in sorted(result_dir.iterdir(), key=lambda path: (step_depth(path.name), path.name)):
        if not step_dir.is_dir():
            continue
        if step_dir.name != "baseline" and not re.fullmatch(r"(?:d|opt-)\d+", step_dir.name):
            continue
        for run_dir in sorted(step_dir.iterdir(), key=lambda path: branch_number(path.name)):
            if not run_dir.is_dir() or not re.fullmatch(r"(?:b|run)\d+", run_dir.name):
                continue
            record_path = run_dir / "record.json"
            if not record_path.exists():
                continue
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(record, dict):
                step = record.get("step", step_dir.name)
                branch = record.get("branch", record.get("run", run_dir.name))
                if isinstance(step, str) and step.startswith("opt-"):
                    step = f"d{step.split('-', 1)[1]}"
                if isinstance(branch, str) and branch.startswith("run"):
                    branch = f"b{branch[3:]}"
                record["step"] = step
                record["branch"] = branch
                records.append(record)
    records.sort(key=lambda record: (step_depth(record.get("step", "")), branch_number(record.get("branch", ""))))
    return records

def extract_solution_path(result_dir: pathlib.Path, record: dict) -> pathlib.Path:
    snapshot_dir = record.get("solution_snapshot_dir")
    if isinstance(snapshot_dir, str) and snapshot_dir:
        candidate = result_dir / snapshot_dir
        if candidate.exists():
            for name in ("kernel.cu", "kernel.py", "binding.py"):
                probe = candidate / name
                if probe.exists():
                    return probe
            return candidate
    return result_dir / "solution"

def build_rows(result_dir: pathlib.Path, records: list[dict]):
    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        rows.append({
            "result_dir": result_dir.name,
            "step": record.get("step", record.get("candidate", "-")),
            "branch": record.get("branch", record.get("run", "-")),
            "parent": record.get("parent") or "-",
            "status": record.get("status", "-"),
            "correctness": record.get("correctness", "-"),
            "source_candidate": "yes" if record.get("source_candidate") else "",
            "latency_ms": metrics.get("latency_ms"),
            "source_latency_ms": record.get("source_latency_ms", record.get("baseline_latency_ms")),
            "delta_ms": (
                metrics.get("latency_ms") - record.get("source_latency_ms", record.get("baseline_latency_ms"))
                if is_number(metrics.get("latency_ms")) and is_number(record.get("source_latency_ms", record.get("baseline_latency_ms")))
                else None
            ),
            "reference_latency_ms": metrics.get("reference_latency_ms"),
            "speedup_factor": metrics.get("speedup_factor"),
            "max_abs_error": metrics.get("max_abs_error"),
            "max_rel_error": metrics.get("max_rel_error"),
            "solution_path": rel(extract_solution_path(result_dir, record)),
        })
    return rows

def print_table(title: str, rows: list[dict], headers: list[tuple[str, str]]):
    print(title)
    if not rows:
        print("(no rows)")
        print()
        return

    rendered = []
    widths = [len(label) for _, label in headers]
    for row in rows:
        cells = []
        for idx, (key, _) in enumerate(headers):
            value = row.get(key)
            if key in {"latency_ms", "source_latency_ms", "delta_ms", "reference_latency_ms", "speedup_factor"}:
                cell = fmt(value)
            elif key in {"max_abs_error", "max_rel_error"}:
                cell = fmt(value, 6)
            else:
                cell = str(value if value is not None else "-")
            cells.append(cell)
            widths[idx] = max(widths[idx], len(cell))
        rendered.append(cells)

    print("  ".join(label.ljust(widths[idx]) for idx, (_, label) in enumerate(headers)))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for cells in rendered:
        print("  ".join(cells[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print()

all_rows = []
current_rows = []

for result_dir in iter_result_dirs(root):
    records = load_step_records(result_dir)
    if not records:
        continue
    rows = build_rows(result_dir, records)
    all_rows.extend(rows)

    current_json_path = result_dir / "current.json"
    current_step = None
    current_branch = None
    if current_json_path.exists():
        try:
            current_data = json.loads(current_json_path.read_text(encoding="utf-8"))
        except Exception:
            current_data = None
        if isinstance(current_data, dict):
            current_step = current_data.get("source_step", current_data.get("current_step"))
            current_branch = current_data.get("source_branch", current_data.get("current_branch", current_data.get("current_run")))

    current = None
    if current_step and current_branch:
        current = next((row for row in rows if row["step"] == current_step and row["branch"] == current_branch), None)
    if current is None:
        current = next((row for row in rows if row["source_candidate"] == "yes"), None)
    if current is None:
        current_rows.append({
            "result_dir": result_dir.name,
            "step": "-",
            "branch": "-",
            "status": "no-source-candidate",
            "correctness": "-",
            "latency_ms": None,
            "reference_latency_ms": None,
            "speedup_factor": None,
            "solution_path": rel(result_dir / "solution"),
        })
    else:
        current_rows.append({
            "result_dir": current["result_dir"],
            "step": current["step"],
            "branch": current["branch"],
            "status": current["status"],
            "correctness": current["correctness"],
            "latency_ms": current["latency_ms"],
            "reference_latency_ms": current["reference_latency_ms"],
            "speedup_factor": current["speedup_factor"],
            "solution_path": current["solution_path"],
        })

print_table(
    "Performance Table",
    all_rows,
    [
        ("result_dir", "result_dir"),
        ("step", "step"),
        ("branch", "branch"),
        ("parent", "parent"),
        ("status", "status"),
        ("correctness", "correctness"),
        ("source_candidate", "source"),
        ("latency_ms", "latency_ms"),
        ("source_latency_ms", "source_ms"),
        ("delta_ms", "delta_ms"),
        ("reference_latency_ms", "ref_ms"),
        ("speedup_factor", "speedup_x"),
        ("max_abs_error", "abs_err"),
        ("max_rel_error", "rel_err"),
        ("solution_path", "solution_path"),
    ],
)

print_table(
    "Source Candidates",
    current_rows,
    [
        ("result_dir", "result_dir"),
        ("step", "step"),
        ("branch", "branch"),
        ("status", "status"),
        ("correctness", "correctness"),
        ("latency_ms", "latency_ms"),
        ("reference_latency_ms", "ref_ms"),
        ("speedup_factor", "speedup_x"),
        ("solution_path", "solution_path"),
    ],
)
PY
