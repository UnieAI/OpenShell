# Kernel Design Agents for OpenShell

This example integrates Kernel Design Agents into OpenShell using a real
FlashInfer-Bench workspace, host bind-mounts, and local result persistence.

The integration is now definition-centric:

1. one workspace maps to one concrete dataset definition
2. preset configs live under `config/presets/`
3. the two core workload profiles are:
   - `w1`: one fixed smoke-test workload UUID
   - `wa`: all workloads for that definition

The MLSys26 dataset currently contains 3 task families and 5 concrete
definitions:

- `moe`
  - `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- `dsa_paged`
  - `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
  - `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
- `gdn`
  - `gdn_decode_qk4_v8_d128_k_last`
  - `gdn_prefill_qk4_v8_d128_k_last`

## Layout

| Path | Purpose |
| --- | --- |
| `Dockerfile` | Image with Codex CLI and the tools needed for draft/execute loops |
| `optimize.sh` | Recommended host-side entrypoint |
| `scaffold.sh` | Copies the starter kit into a writable workspace and adds KDA files |
| `config/presets/` | Definition-centric preset configs |
| `config/kda-gemm-task*.yml` | Legacy MoE compatibility presets |
| `starter-kit/` | Vendored `flashinfer-bench-starter-kit` template |
| `scripts/run-kda-draft.sh` | Container-side KDA runner for both draft and execute modes |

## Preset Model

Preset naming is:

```text
config/presets/<family>/<definition>-<profile>.yml
```

Examples:

- `config/presets/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048-w1.yml`
- `config/presets/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64-wa.yml`
- `config/presets/gdn/gdn_decode_qk4_v8_d128_k_last-w1.yml`

Each preset fixes:

- the concrete `definition`
- the local result `workspace`
- the benchmark scope
- the task contract defaults
- the starter-kit build metadata

Core profiles:

- `w1`
  uses one fixed UUID from the dataset for smoke validation
- `wa`
  leaves `benchmark_workload_uuids` empty and runs all workloads for that definition

Legacy note:

- `config/kda-gemm-task.yml`
  remains the default entry path for compatibility and now behaves as a MoE `w1` preset
- `config/kda-gemm-task-1.yml`
  remains a legacy explicit MoE `w1` preset
- `config/kda-gemm-task-4.yml`
  is kept only as a legacy custom UUID subset and is not part of the new generic core model

## Recommended Flow

Run from the OpenShell repo root.

Draft-only:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/kernel-design-agents/optimize.sh \
  --config=examples/kernel-design-agents/config/presets/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048-w1.yml \
  --gpus=1
```

Execute:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/kernel-design-agents/optimize.sh \
  --config=examples/kernel-design-agents/config/presets/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048-w1.yml \
  --mode=execute \
  --gpus=1
```

`--mode=execute` without `--kernel-optimize` means one baseline smoke /
baseline revalidation attempt only. It does not enter an optimization loop.

Kernel optimization:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/kernel-design-agents/optimize.sh \
  --config=examples/kernel-design-agents/config/presets/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048-w1.yml \
  --mode=execute \
  --kernel-optimize \
  --max-depth=2 \
  --branch=2 \
  --gpus=1
```

`--kernel-optimize` means:

- if no source candidate exists in workspace results, bootstrap a baseline first
- if a source candidate already exists, skip baseline bootstrap and optimize from that source
- then try up to `--max-depth=<n>` promoted depth levels
- at each depth, allow up to `--branch=<n>` new candidate attempts before giving up on that depth

This is intended to push Codex effort toward `solution/`, especially
`solution/cuda/kernel.cu`, instead of spending the run on orchestration churn.

## Workspace Semantics

The runner does the following:

1. scaffold a writable workspace from `starter-kit/` when needed
2. write `TASK_CONTRACT.md` and `config.toml` from the flat config
3. bind-mount the workspace into Docker
4. run either a draft-only or implementation/validation loop
5. persist plans, summaries, logs, and benchmark outputs under the host workspace

Typical outputs:

- `docs/draft.md` in draft mode
- `outputs/execution-summary.agent.md`
- `outputs/last-message.md`
- `outputs/execution-summary.md`
- `runs/run_local.txt`
- `runs/run_local_results.json`
- `benchmark.csv`
- `candidates.jsonl`
- `baseline/b*/...` and `d*/b*/...` per-attempt workspaces in execute mode

`benchmark.csv` and `candidates.jsonl` are orchestration-generated exports from
the immutable step/branch records. The agent should not treat them as control
inputs.
Each execute attempt runs in a fresh immutable workspace. Earlier baseline or
depth branches are preserved in place and are not restored over or mutated.

Result directories are isolated per preset because each preset has its own
workspace path, for example:

- `results/kda-moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048-w1`
- `results/kda-moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048-wa`
- `results/kda-dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64-w1`

## Build Defaults

The current generic presets all start from the same OpenShell submission path:

- `language: cuda`
- `entry_point: kernel.cu::kernel`
- `source_dir: cuda`
- `binding: torch`

The runner keeps CUDA + `binding=torch` on `kernel.cu::kernel` and will not
silently switch the active benchmark path to `binding.py::...`.

## Dataset

If the config sets:

```text
fib_dataset_path: "examples/kernel-design-agents/mlsys26-contest"
```

then execute mode mounts that host path read-only into the container and
exports `FIB_DATASET_PATH` automatically.

`run_local.py` requires `FIB_DATASET_PATH` for benchmarking. Draft mode does
not need the dataset. Execute mode can still validate packaging without it, but
evaluation will be recorded as blocked.

## Preset Catalog

The generic preset set currently covers all 5 concrete definitions:

- `config/presets/moe/`
  - `...-w1.yml`
  - `...-wa.yml`
- `config/presets/dsa_paged/`
  - `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64-w1.yml`
  - `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64-wa.yml`
  - `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64-w1.yml`
  - `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64-wa.yml`
- `config/presets/gdn/`
  - `gdn_decode_qk4_v8_d128_k_last-w1.yml`
  - `gdn_decode_qk4_v8_d128_k_last-wa.yml`
  - `gdn_prefill_qk4_v8_d128_k_last-w1.yml`
  - `gdn_prefill_qk4_v8_d128_k_last-wa.yml`

If you want a custom subset such as representative UUIDs, keep using
`benchmark_workload_uuids` directly in a dedicated config. That is now treated
as an optional custom profile layer, not a core preset type.

## Useful Flags

- `--mode=draft|execute`
- `--kernel-optimize`
- `--max-depth=<n>`
- `--branch=<n>`
- `--clean-workspace`
- `--fib-dataset-path=<path>`
- `--benchmark-workload-uuids=<uuid1,uuid2,...>`

## Smoke Test

```shell
cd OpenShell
bash examples/kernel-design-agents/smoke.sh --gpus=1
```

This checks image startup, packaged tools, scaffold behavior, and draft-runner
availability without calling the model API.
