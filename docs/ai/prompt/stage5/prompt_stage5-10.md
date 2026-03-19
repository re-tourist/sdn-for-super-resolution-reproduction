You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.10:

Issue title:
Run paper-aligned main training (phase-only, L=5) and write report

Important execution constraint:

- The real full-budget main run will be executed later on a Linux server.
- In this workspace, do NOT launch a long full-dataset main run locally.
- Your job here is to make the main run launch-ready:
  - finalize the main-run config/code path
  - write the exact Linux-server run commands into `docs/run_order.md`
  - add any narrow report/template docs needed for later result collection

This is a Stage-5-scoped main-run-prep task under a remote-execution constraint.
Do not turn it into a local long-run execution task or a Stage 6 sweep.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
4. `docs/plan/stage_plan/stage5/stage5_plan.md`
5. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
6. `docs/execution/stage5_smoke_report.md`
7. `docs/run_order.md`
8. `configs/stage5/stage5_trainer_smoke.yaml`
9. `configs/stage5/stage5_eval.yaml`
10. `configs/stage5/stage5_blind_eval.yaml`

Then inspect these implementation-reference files before editing:

11. `scripts/train_stage5_paper.py`
12. `scripts/eval_stage5_paper.py`
13. `scripts/eval_stage5_blind_linepair.py`
14. `docs/ai/CodexFeedback/stage5/feedback_stage5-9.md`

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- This issue should prepare the main run, not silently execute it locally.
- The deliverable must be launch-ready for Linux-server execution.

==================================================
1. ISSUE 5.10 GOAL
==================================================

Prepare the paper-aligned main-run path for phase-only `L=5` so that the user
can execute it on a Linux server with clear commands and traceable artifacts.

The result should provide:

1. a dedicated main-run config
2. exact Linux-server training / eval / blind-eval commands in `docs/run_order.md`
3. a clear separation between:
   - code/config path is ready
   - full long-run results are pending server execution
4. any minimal report scaffold needed for later result capture

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not launch a real long full-budget main run locally in this workspace
- do not rewrite trainer / eval / blind-eval architecture
- do not reopen frozen Stage 5 protocol items
- do not add Stage 6 ablations, robustness, or quantization work
- do not present smoke results as main-run results
- do not hide the Linux-server constraint

This issue is successful only if it makes the main run ready to launch on the
Linux server and documents that launch path clearly.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/train`

Stay on a training/execution-oriented branch.
Do not absorb Stage 5.11 doc-consolidation scope beyond what this issue strictly needs.

==================================================
4. REQUIRED INHERITANCE
==================================================

From the Stage 5 docs and prior issues, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Smoke run already demonstrated local pipeline stability.
3. Regular eval + bicubic baseline already exist from Issue 5.7.
4. Blind eval hook already exists from Issue 5.8.
5. Main run should consume those paths, not redesign them.

Important inheritance detail:

- The main-run target is phase-only `L=5`.
- The actual heavy execution belongs on the Linux server.
- This issue must therefore emphasize launch readiness, command clarity, and
  artifact conventions over local long-run results.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the dedicated Stage 5 main-run config
2. the exact Linux-server command sequence in `docs/run_order.md`
3. any narrow main-run naming / output-root convention
4. a report template or placeholder doc for later server results

This issue MUST keep the following explicit and auditable:

1. main-run config path
2. expected output root
3. training command
4. resume command
5. regular-eval command
6. blind-eval command
7. what is prepared now vs what must happen later on Linux

This issue MUST NOT finalize:

1. actual long-run result claims without execution evidence
2. Stage 6 experimental scope
3. new trainer semantics

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real main-run-prep path needed for Linux-server launch.

Prefer a narrow structure such as:

- `configs/stage5/stage5_trainer_main.yaml`
- updates to `docs/run_order.md`
- optional `docs/execution/stage5_main_run_report.md` as a template / placeholder

The implementation should likely include:

- a dedicated main-run config for phase-only `L=5`
- explicit output-root and run-name convention
- exact Linux commands for:
  - training
  - resuming
  - regular eval
  - blind eval
- any minimal local dry-run or config-validation step if cheap enough

If a very small local validation is useful, keep it strictly lightweight:

- config parse / script help / one-step dry-run only if needed
- no hidden full main run

==================================================
7. RUN-ORDER REQUIREMENT
==================================================

Because the user will run the full main training on a Linux server, you must
write the operational command recipe into:

- `docs/run_order.md`

This is required.

The added section should be easy to follow and should include:

1. the main training command
2. the resume command
3. the regular-eval command
4. the blind-eval command
5. expected output directories
6. any first-run notes such as dataset download behavior or GPU device choice

Do NOT hide the commands only in `README.md`.
`docs/run_order.md` is the required operational handoff document for this issue.

==================================================
8. REQUIRED ARTIFACTS
==================================================

At minimum, provide:

- a dedicated main-run config under `configs/stage5/`
- an updated `docs/run_order.md` with Linux-server commands
- optionally a main-run report template doc if useful

If you add a report template, it must clearly say that:

- this workspace prepared the launch path
- full results are pending actual Linux-server execution

==================================================
9. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- `configs/stage5/stage5_trainer_main.yaml`
- `docs/run_order.md`
- optionally `docs/execution/stage5_main_run_report.md`

Supporting edits are allowed in:

- `README.md`

but only if a short reference is helpful. The authoritative operational commands
must still go into `docs/run_order.md`.

Avoid:

- broad code refactors
- local long-run execution artifacts
- smoke-report rewrites

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 5.10 is complete only if:

1. A real main-run config for phase-only `L=5` exists
2. `docs/run_order.md` contains clear Linux-server commands for train / resume / eval
3. The repo clearly distinguishes launch readiness from actual executed results
4. No frozen Stage 5 semantics are silently changed
5. The handoff is sufficient for the user to run the main job on the Linux server

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was prepared
- exact files changed
- the main-run config path
- the exact sections/commands added to `docs/run_order.md`
- any lightweight local validation that was performed
- what remains to be executed later on the Linux server
