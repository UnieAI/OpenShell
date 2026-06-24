# Kernel Design Agents for OpenShell

This example integrates the early Kernel Design Agents loop into OpenShell,
using a real task workspace instead of a fake GEMM skeleton.

The packaged workflow is now:

1. vendor `flashinfer-bench-starter-kit` under this example
2. derive a writable task workspace under `examples/kernel-design-agents/results/`
3. write `TASK_CONTRACT.md` and `config.toml` from a flat YAML config
4. run either a draft-only or implementation loop in Docker against that host workspace
5. keep the resulting plans, summaries, and final message on disk

This stays close to `examples/autoagent`: Docker on the host, direct bind
mounts, and results persisted locally.

## Layout

| Path | Purpose |
| --- | --- |
| `Dockerfile` | Image with Codex CLI and the tools needed for the draft loop |
| `optimize.sh` | Recommended single-entry runner |
| `scaffold.sh` | Copies the vendored starter kit into a writable workspace and adds KDA files |
| `config/kda-gemm-task.yml` | Flat config for the default FlashInfer task preset |
| `starter-kit/` | Vendored `flashinfer-bench-starter-kit` template |
| `scripts/run-kda-draft.sh` | Container-side KDA runner for both draft and execute modes |
| `plan.sh` | Older OpenShell sandbox path, kept as an alternative |
| `smoke.sh` | OpenShell sandbox smoke test for the packaged image and helper scripts |

## Recommended Flow

Run the single-entry script from the OpenShell repo root.

Draft-only mode:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/kernel-design-agents/optimize.sh \
  --config=examples/kernel-design-agents/config/kda-gemm-task.yml \
  --gpus=1
```

What it does:

1. copies `examples/kernel-design-agents/starter-kit` into
   `examples/kernel-design-agents/results/kda-flashinfer-moe-phase1` when needed
2. writes `TASK_CONTRACT.md` from the YAML config
3. writes `config.toml` with the configured solution metadata
4. builds or reuses `openshell-kda-example`
5. bind-mounts the workspace into Docker
6. runs the draft-only Codex loop

Outputs land here by default:

- `examples/kernel-design-agents/results/kda-flashinfer-moe-phase1/docs/draft.md`
- `examples/kernel-design-agents/results/kda-flashinfer-moe-phase1/outputs/last-message.md`

Implementation mode:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/kernel-design-agents/optimize.sh \
  --config=examples/kernel-design-agents/config/kda-gemm-task.yml \
  --mode=execute \
  --fib-dataset-path=/path/to/mlsys26-contest \
  --gpus=1 \
  --build
```

What execute mode adds:

1. keeps `docs/draft.md` and updates `docs/plan.md`
2. allows Codex to edit the workspace implementation
3. runs the contract validation command inside the container
4. exports reduced-benchmark defaults and log paths into the container
5. runs the evaluation command when `FIB_DATASET_PATH` is mounted
6. expects a concise execution summary in
   `examples/kernel-design-agents/results/kda-flashinfer-moe-phase1/outputs/execution-summary.md`

If a baseline already validates and you want the run to focus on latency work,
add `--kernel-optimize`. That tells the runner to steer Codex toward
kernel-level performance changes under `solution/` instead of spending the run
on baseline bring-up or integration churn. If the workspace does not yet have a
validated baseline, the runner first bootstraps one and then asks Codex to
attempt optimization candidate(s) in the same run. Use `--max-depth=<n>` to
cap how many optimization candidates the run may attempt; the default is `1`.

## Default Preset

The default config still lives at
`examples/kernel-design-agents/config/kda-gemm-task.yml` for compatibility, but
it now describes an official-style FlashInfer MoE phase-1 task:

- definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- language: `cuda`
- entry point: `kernel.cu::kernel`
- source dir: `cuda`
- destination passing style: `false`
- binding: `torch`
- reduced benchmark defaults:
  `warmup_runs=1`, `iterations=3`, `num_trials=1`, `workload_limit=2`
- validation: `python scripts/pack_solution.py`
- evaluation:
  `FIB_DATASET_PATH=/path/to/mlsys26-contest python scripts/run_local.py`

`run_local.py` requires `FIB_DATASET_PATH`. The draft-only loop does not need
the dataset to exist. Execute mode can still validate without it, but
evaluation will be treated as blocked unless `--fib-dataset-path` or
`FIB_DATASET_PATH` is provided.

`run_local.py` now supports KDA-friendly staged execution:

- it can limit the benchmark to a reduced workload subset
- it can lower benchmark repetitions for execute-mode smoke runs
- it auto-disables trace dumping when the mounted dataset path is read-only
- it mirrors stdout/stderr to `runs/run_local.txt`
- it writes structured results to `runs/run_local_results.json`

For CUDA + `binding=torch` workspaces, `scripts/check_cuda_extension.py` is also
available as a direct compile-smoke step before running the benchmark.
Keep that path on a CUDA entry point such as `kernel.cu::kernel`; `binding.py`
is not the active benchmark entry point for the torch-extension flow.

The default contract also assumes FlashInfer contest submission rules:

- final candidate code must live under `solution/`
- the submission must stay self-contained
- runtime wrappers around `flashinfer`, `deep_gemm`, and similar specialized
  kernel libraries are not acceptable candidates
- dataset baseline solutions are references for semantics and benchmarking
  context, not submission implementations

## Manual Scaffold

To create the workspace without running Codex yet:

```shell
cd OpenShell
bash examples/kernel-design-agents/scaffold.sh \
  examples/kernel-design-agents/results/kda-flashinfer-moe-phase1
```

That workspace includes:

- the full FlashInfer starter-kit files such as `config.toml`, `solution/`, and `scripts/`
- KDA bookkeeping files: `TASK_CONTRACT.md`, `docs/draft.md`, `docs/plan.md`
- evidence/output directories: `runs/`, `outputs/`, `profile/`

If you scaffold manually, edit at least:

- `TASK_CONTRACT.md`
- `config.toml`
- `solution/triton/kernel.py` or `solution/cuda/*`

## Config Overrides

The runner accepts overrides on top of the YAML config, for example:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/kernel-design-agents/optimize.sh \
  --config=examples/kernel-design-agents/config/kda-gemm-task.yml \
  --workspace=examples/kernel-design-agents/results/kda-sparse-attention \
  --definition=sparse_attention \
  --solution-name=openshell-kda-sparse-v1 \
  --gpus=1
```

Useful override flags:

- `--workspace=...`
- `--starter-kit=...`
- `--mode=draft|execute`
- `--fib-dataset-path=...`
- `--solution-name=...`
- `--definition=...`
- `--author=...`
- `--language=triton|cuda`
- `--entry-point=...`
- `--source-dir=...`
- `--destination-passing-style=true|false`
- `--binding=tvm-ffi|torch`
- `--benchmark-warmup-runs=...`
- `--benchmark-iterations=...`
- `--benchmark-num-trials=...`
- `--benchmark-workload-limit=...`
- `--benchmark-workload-uuids=uuid1,uuid2`
- `--kernel-optimize`
- `--max-depth=<n>`
- `--clean-workspace`

Preset configs:

- `config/kda-gemm-task-1.yml`
  uses one fixed smoke-test workload UUID from the official dataset
- `config/kda-gemm-task-4.yml`
  uses four fixed representative workload UUIDs spanning `seq_len=1,32,901,14107`

## Smoke Test

To verify the packaged example image and helper scripts through OpenShell:

```shell
cd OpenShell
bash examples/kernel-design-agents/smoke.sh --gpus=1
```

This does not call the model API. It checks that:

- the image starts in an OpenShell sandbox
- the packaged tools are present
- Python runtime can import `flashinfer_bench` and `modal`
- `scaffold.sh` produces a starter-kit-backed workspace
- the draft runner is callable

## Alternative OpenShell Sandbox Path

`plan.sh` still exists if you want the older upload/exec/download pattern:

```shell
cd OpenShell
bash examples/kernel-design-agents/plan.sh \
  --workspace examples/kernel-design-agents/results/kda-flashinfer-moe-phase1 \
  --gpus=1
```

The recommended path is still `optimize.sh`, because it matches `autoagent`
more closely, supports dataset mounting, and avoids host/container file
transfer issues.
