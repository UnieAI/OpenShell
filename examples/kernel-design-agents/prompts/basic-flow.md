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

## Inspection Rules

- Do not infer implementation details from the repository name, template, or
  prior expectations alone.
- Do not claim a file's contents, active build path, kernel signature, or
  validation flow unless you read the relevant file in this run.
- If the workspace contains build/config files, implementation files, or run
  scripts, inspect those exact files before writing the draft.
- When a workspace includes multiple implementation paths such as Triton and
  CUDA, explicitly identify which one is active from the local config files.

## Plan Draft Requirements

The draft in `docs/draft.md` should include:

- The current baseline and how it is validated
- The main risks and unknowns
- Candidate implementation directions ranked by expected value and risk
- The first concrete implementation steps
- The exact validation and evaluation commands to run
- The evidence required to promote, revise, or reject a candidate
- Short file references for the baseline claims that matter most

Do not start implementation until the draft exists.
