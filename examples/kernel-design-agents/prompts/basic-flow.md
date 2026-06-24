# Kernel Design Agents Basic Flow Prompt

You are working in a task implementation workspace. Your job is to produce the
best correct implementation for the task described below.

## Task Contract

Read `TASK_CONTRACT.md` in the current workspace. Treat that file as the source
of truth for:

- Task name
- Objective
- Correctness requirements
- Performance or quality target
- Allowed implementation approaches
- Validation command
- Evaluation command
- Promotion criteria

## Workflow

1. Read the repository structure, existing implementation, tests, and task
   documentation.
2. Identify the baseline behavior and the validation path.
3. Research only the references needed for this task.
4. Write an implementation-plan draft to `docs/draft.md`.
5. Turn the draft into an executable plan before editing code.
6. Implement one candidate at a time.
7. Run validation after each meaningful candidate.
8. Record candidate results, parent relationships, and evidence in the
   workspace.
9. Keep the final change scoped to the task contract.

For expensive kernels, prefer staged validation over all-at-once benchmarking:
pack first, then any direct build/compile smoke, then a reduced benchmark
subset, and only then a broader sweep if the candidate still looks viable.
If the workspace provides `scripts/check_cuda_extension.py` and the active
build is CUDA with `binding=torch`, use that script for the compile-smoke step.
When you run `scripts/run_local.py`, keep the benchmark artifacts in the
canonical paths `runs/run_local.txt` and `runs/run_local_results.json`.

## Inspection Rules

- Do not infer implementation details from the repository name, template, or
  prior expectations alone.
- Do not claim a file's contents, active build path, kernel signature, or
  validation flow unless you read the relevant file in this run.
- If the active build path is CUDA, identify the configured binding mode from
  the local config before choosing between TVM FFI and Torch-extension style
  implementations.
- For CUDA with `binding=torch`, keep the callable entry point on the CUDA
  source file, for example `kernel.cu::kernel`. Do not switch that build path
  to `binding.py::...`; `binding.py` is not the active benchmark entry point
  for this torch-extension flow.
- If the workspace contains build/config files, implementation files, or run
  scripts, inspect those exact files before writing the draft.
- When a workspace includes multiple implementation paths such as Triton and
  CUDA, explicitly identify which one is active from the local config files.
- Treat benchmark baselines, packaged solutions, and dataset artifacts as
  references for semantics, constraints, and performance context. Do not turn
  them into runtime dependencies unless the task contract explicitly allows it.
- Prefer self-contained implementation changes under the active submission
  source directory. Do not mistake harness-only or docs-only edits for task
  completion.

## Plan Draft Requirements

The draft in `docs/draft.md` should include:

- The current baseline and how it is validated
- The main risks and unknowns
- Candidate implementation directions ranked by expected value and risk
- The first concrete implementation steps
- The exact validation and evaluation commands to run
- Which reduced benchmark scope you will use before any broader sweep
- The evidence required to promote, revise, or reject a candidate
- Short file references for the baseline claims that matter most

Do not start implementation until the draft exists.
