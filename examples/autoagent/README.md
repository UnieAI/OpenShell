# AutoAgent for OpenShell

This example vendors the original `AutoAgent` workflow into the OpenShell repo
and repackages it as a local BYOC sandbox example.

The current OpenShell-compatible image is intentionally narrow:

- Supported: render, compile, verify, `--test-only`
- Not enabled by default: `--profile`, `--optimize`

This is a single image with capability-gated execution paths. The entrypoint
probes runtime requirements up front and fails fast with a clear error when a
mode is unavailable.

For the supported compile-and-verify path, the effective runtime dependencies
are:

- `torch`
- `apache-tvm-ffi`
- `PyYAML`
- `Jinja2`

The vendored tree still carries additional Python dependencies in
`pyproject.toml` for broader upstream compatibility, but `torchvision` and
`torchaudio` are not used by the current OpenShell compile-and-verify flow.

Profiling is split behind a dedicated executor because standard OpenShell
sandboxes block `ptrace` and `perf_event_open`, which Nsight Compute depends on.

For regression coverage, the integration now also includes an example-local
smoke script that validates the single image contents, the compile path, and
the `--test-only` path in one sandbox.

The optimization path now also has a dedicated example-local runner for
profiling-capable environments:

- `smoke.sh`: OpenShell sandbox compile/verify smoke
- `optimize.sh`: dedicated Docker runner for baseline profiling and tree search

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Example entrypoint |
| `config.yaml` | Runtime settings for paths, backend, and profile executor |
| `configs/` | Kernel problem definitions |
| `tools/` | Pipeline, backend, runtime, and profiling helpers |
| `sandbox-policy.yaml` | Minimal offline policy for compile and verify |

## End-to-end setup

The commands below describe the current known-good bring-up flow on a Docker
host with NVIDIA GPUs.

### 1. Install OpenShell CLI

```shell
cd OpenShell
python3 -m venv .venv
source .venv/bin/activate
uv tool install -U .
```

### 2. Generate the gateway JWT keypair

The Docker deployment expects a signing key, public key, and key id under
`/var/lib/openshell/jwt`.

```shell
sudo mkdir -p /var/lib/openshell/jwt
cd /var/lib/openshell/jwt

sudo openssl genpkey -algorithm Ed25519 -out signing.pem
sudo openssl pkey -in signing.pem -pubout -out public.pem
openssl rand -hex 16 | sudo tee kid >/dev/null

sudo chmod 600 signing.pem
sudo chmod 644 public.pem kid
```

### 3. Start the local Docker gateway

```shell
cd OpenShell/deploy/docker
docker compose up
curl -sf http://localhost:8080/healthz
```

Register the gateway with the CLI:

```shell
openshell gateway add http://localhost:8080 --name local
openshell gateway select local
openshell status
```

If `openshell sandbox create` reports `No active gateway`, the `gateway add`
and `gateway select` steps above were not completed successfully.

### 4. Smoke-test the packaged AutoAgent image

This validates the packaged Python dependencies plus the compile, verify, and
`--test-only` flows inside a standard OpenShell sandbox.

```shell
cd OpenShell
bash examples/autoagent/smoke.sh --gpus=1
```

### 5. Run kernel optimization

For tree search, the current runner expects exactly one auth env:

- `OPENAI_API_KEY`
- `CODEX_API_KEY`

Set only one of them. `optimize.sh` mirrors the provided key into the env names
that the in-container Codex CLI expects. Do not set both.

Example with `OPENAI_API_KEY`:

```shell
cd OpenShell
OPENAI_API_KEY='sk-...' \
bash examples/autoagent/optimize.sh --gpus=1 --max-depth=1 --branch-factor=1
```

### 6. Check the optimization result

Expected terminal output looks like:

```text
========================================================================================================================
TREE SEARCH RESULTS
========================================================================================================================
Label      Duration(us)   MemThpt(%)   CompThpt(%)   Occupancy(%)   SMBusy(%)   Mem(GB/s)   Description
------------------------------------------------------------------------------------------------------------------------
d0/b0      135.87         2.77         2.77          2.08           2.02        3.93        baseline
d1/b0      5.28           7.17         4.62          16.08          3.72        100.61      codex-exec d1 branch 0
========================================================================================================================
```

The summary CSV is written to:

```shell
ls OpenShell/examples/autoagent/results/kernels/results.csv
```

## Build and run with OpenShell

Create a GPU sandbox from this directory and run a compile plus verify pass:

```shell
openshell sandbox create \
  --from examples/autoagent \
  --gpu \
  --policy examples/autoagent/sandbox-policy.yaml \
  -- /app/.venv/bin/python /app/main.py --config configs/gemm.yaml
```

Run verification only against a previously generated shared object and active
kernel metadata:

```shell
openshell sandbox create \
  --from examples/autoagent \
  --gpu \
  --policy examples/autoagent/sandbox-policy.yaml \
  -- /app/.venv/bin/python /app/main.py --config configs/gemm.yaml --test-only
```

The image writes generated files under `/sandbox/autoagent` by default through
`AUTOAGENT_STATE_DIR`, so the source tree under `/app` stays read-only.
The image also wires `python` and `python3` to the bundled `/app/.venv`
interpreter at build time, and the build fails if `import tvm_ffi` does not
work through that command path. At runtime, the following capability checks are
applied before execution:

- `compile`: `torch`, `tvm_ffi`, `nvcc`
- `verify` / `--test-only`: `torch`, `tvm_ffi`
- `--profile`: `torch`, `tvm_ffi`, `nvcc`, `ncu`
- `--optimize --max-depth 0`: `torch`, `tvm_ffi`, `nvcc`, `ncu`
- `--optimize` with `--max-depth > 0`: `torch`, `tvm_ffi`, `nvcc`, `ncu`, `codex`

To run the example smoke against a Docker-backed GPU gateway:

```shell
bash examples/autoagent/smoke.sh --gpus=1
```

Useful overrides:

- `bash examples/autoagent/smoke.sh --gpus=1`
- `AUTOAGENT_KEEP_SANDBOX=1`
- `AUTOAGENT_CONFIG=configs/gemm.yaml`
- `AUTOAGENT_STATE_DIR=/sandbox/autoagent`

## Optimization bring-up

`--optimize` has two practical stages:

1. Baseline profiling only:
   `--optimize --max-depth 0`
   This still renders, compiles, verifies, profiles the baseline kernel, and
   emits `kernels/results.csv`, but it does not call Codex.
2. Full tree search:
   `--optimize --max-depth 1 --branch-factor 1` or higher
   This additionally requires a working Codex CLI inside the profiling
   container.

The dedicated runner uses Docker directly instead of `openshell sandbox create`
because Nsight Compute generally needs relaxed container privileges such as
`SYS_ADMIN`, `SYS_PTRACE`, and `seccomp=unconfined`. It also defaults to
running the profiling container as `root` so GPU performance counters work on
hosts that restrict them to admin users. The optimize runner is image-first: it
executes the code baked into the image, not a host source mount. Rebuild the
image after Python, shell, or Dockerfile changes.

Preflight the profiling environment:

```shell
bash examples/autoagent/optimize.sh --build --check --gpus=1
```

After changes to `main.py`, `tools/*.py`, `optimize.sh`, or `Dockerfile`,
rebuild before running optimize:

```shell
bash examples/autoagent/optimize.sh --build --gpus=1 --max-depth=0
```

Produce a baseline `results.csv` for `softmax.yaml`:

```shell
bash examples/autoagent/optimize.sh --build --gpus=1 --max-depth=0
```

Try a minimal full search once NCU and Codex are both available:

```shell
OPENAI_API_KEY='sk-...' \
bash examples/autoagent/optimize.sh \
  --gpus=1 --config=configs/softmax.yaml --max-depth=1 --branch-factor=1
```

If you need to override the profiling user explicitly:

```shell
bash examples/autoagent/optimize.sh --gpus=1 --docker-user=0:0
```

Expected host-side outputs under `examples/autoagent/results/` by
default:

- `kernels/results.csv`
- `kernels/d0/b0/*`
- `kernels/d1/b0/*` when tree search produces a depth-1 candidate
- `output/best_kernel.yaml`
- `output/best_kernel.cu`
- `output/best_kernel.so`

## Runtime configuration

The example no longer assumes `/workspace` or the current working directory.

Available overrides:

- `--workspace` or `AUTOAGENT_WORKSPACE`
- `--state-dir` or `AUTOAGENT_STATE_DIR`
- `--runtime-config` or `AUTOAGENT_RUNTIME_CONFIG`
- `--backend` or `AUTOAGENT_BACKEND`
- `--profile-executor` or `AUTOAGENT_PROFILE_EXECUTOR`
- `AUTOAGENT_PYTHON_COMMAND`
- `AUTOAGENT_CODEX_COMMAND`
- `AUTOAGENT_NCU_COMMAND`

Relative kernel config paths are resolved against the configured workspace.
Generated outputs are resolved against the configured state directory unless an
absolute path is provided in `config.yaml`.

## Backends

Kernel generation now goes through a backend interface.

- `codex-exec`: preserves the original `codex exec` flow

The example image does not assume Codex CLI is present on the standard
OpenShell sandbox path. For full tree search, the example image now installs
Codex CLI directly. `AUTOAGENT_CODEX_COMMAND` remains available if you need to
override the in-image command with a different install.

## Profiling executors

Profiling now runs through an explicit executor interface instead of being hard
wired into the compile pipeline.

- `disabled`: default for OpenShell sandboxes
- `ncu`: local Nsight Compute executor for dedicated profiling environments

If you need profiling, run the same source tree in an environment where NCU is
installed and seccomp policy allows it, then invoke:

```shell
python3 main.py \
  --config configs/gemm.yaml \
  --profile \
  --profile-executor ncu
```
