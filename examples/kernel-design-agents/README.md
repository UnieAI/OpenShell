# Kernel Design Agents for OpenShell

This example integrates the early Kernel Design Agents draft loop into
OpenShell, but uses a real task workspace instead of a fake GEMM skeleton.

The packaged workflow is now:

1. vendor `flashinfer-bench-starter-kit` under this example
2. derive a writable task workspace under `examples/kernel-design-agents/results/`
3. write `TASK_CONTRACT.md` and `config.toml` from a flat YAML config
4. run a draft-only Codex pass in Docker against that host workspace
5. keep the resulting `docs/draft.md` and `outputs/last-message.md` on disk

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
| `scripts/run-kda-draft.sh` | Container-side draft runner |
| `plan.sh` | Older OpenShell sandbox path, kept as an alternative |
| `smoke.sh` | OpenShell sandbox smoke test for the packaged image and helper scripts |

## Recommended Flow

Run the single-entry script from the OpenShell repo root:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/kernel-design-agents/optimize.sh \
  --config=examples/kernel-design-agents/config/kda-gemm-task.yml \
  --gpus=1
```

What it does:

1. copies `examples/kernel-design-agents/starter-kit` into
   `examples/kernel-design-agents/results/kda-flashinfer-task` when needed
2. writes `TASK_CONTRACT.md` from the YAML config
3. writes `config.toml` with the configured solution metadata
4. builds or reuses `openshell-kda-example`
5. bind-mounts the workspace into Docker
6. runs the draft-only Codex loop

Outputs land here by default:

- `examples/kernel-design-agents/results/kda-flashinfer-task/docs/draft.md`
- `examples/kernel-design-agents/results/kda-flashinfer-task/outputs/last-message.md`

## Default Preset

The default config still lives at
`examples/kernel-design-agents/config/kda-gemm-task.yml` for compatibility, but
it now describes a real FlashInfer starter-kit task:

- definition: `fused_moe`
- language: `triton`
- entry point: `kernel`
- validation: `python scripts/pack_solution.py`
- evaluation:
  `FIB_DATASET_PATH=/path/to/mlsys26-contest python scripts/run_local.py`

`run_local.py` requires `FIB_DATASET_PATH`. The draft-only loop does not need
the dataset to exist, but any real validation or benchmarking does.

## Manual Scaffold

To create the workspace without running Codex yet:

```shell
cd OpenShell
bash examples/kernel-design-agents/scaffold.sh \
  examples/kernel-design-agents/results/kda-flashinfer-task
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
- `--solution-name=...`
- `--definition=...`
- `--author=...`
- `--language=triton|cuda`
- `--entry-point=...`
- `--force-scaffold`

## Smoke Test

To verify the packaged example image and helper scripts through OpenShell:

```shell
cd OpenShell
bash examples/kernel-design-agents/smoke.sh --gpus=1
```

This does not call the model API. It checks that:

- the image starts in an OpenShell sandbox
- the packaged tools are present
- `scaffold.sh` produces a starter-kit-backed workspace
- the draft runner is callable

## Alternative OpenShell Sandbox Path

`plan.sh` still exists if you want the older upload/exec/download pattern:

```shell
cd OpenShell
bash examples/kernel-design-agents/plan.sh \
  --workspace examples/kernel-design-agents/results/kda-flashinfer-task \
  --gpus=1
```

The recommended path is still `optimize.sh`, because it matches `autoagent`
more closely and avoids host/container file transfer issues.
