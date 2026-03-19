You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.9:

Issue title:
Run paper-aligned smoke training and write Stage 5 smoke report

This is a Stage-5-scoped smoke-run task.
Do not turn it into main-run tuning, a trainer rewrite, or a Stage 6 experiment sweep.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
4. `docs/plan/stage_plan/stage5/stage5_plan.md`
5. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
6. `configs/stage5/stage5_emnist_display.yaml`
7. `configs/stage5/stage5_optics_paper_aligned.yaml`
8. `configs/stage5/stage5_paper_encoder.yaml`
9. `configs/stage5/stage5_sr_loss.yaml`
10. `configs/stage5/stage5_trainer_short.yaml`
11. `configs/stage5/stage5_eval.yaml`
12. `configs/stage5/stage5_blind_eval.yaml`

Then inspect these implementation-reference files before editing:

13. `scripts/train_stage5_paper.py`
14. `scripts/eval_stage5_paper.py`
15. `scripts/eval_stage5_blind_linepair.py`
16. `docs/ai/CodexFeedback/stage5/feedback_stage5-6.md`
17. `docs/ai/CodexFeedback/stage5/feedback_stage5-7.md`
18. `docs/ai/CodexFeedback/stage5/feedback_stage5-8.md`

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- This issue consumes the already-landed Stage 5 pipeline.
- This issue does not reopen dataset, optics, encoder, loss, trainer, regular eval,
  or blind-eval design questions unless a narrow blocker fix is strictly required.

==================================================
1. ISSUE 5.9 GOAL
==================================================

Run one short paper-aligned smoke training and write a concise report that
confirms the end-to-end Stage 5 pipeline is stable and inspectable.

The result should provide:

1. a real smoke-run output directory under `outputs/stage5/`
2. evidence that the Stage 5 path can train without NaN / Inf
3. concise reporting of loss trend, artifacts, and evaluation hooks
4. a clear boundary between smoke validation and later main-run work

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not treat smoke as the final paper result
- do not expand this into main-run tuning or hyperparameter search
- do not rewrite the Stage 5 trainer unless a narrow blocker fix is unavoidable
- do not reopen frozen Stage 5 protocol items
- do not add Stage 6 ablations, robustness, or quantization work
- do not silently replace the smoke budget with a long-run budget

This issue is successful only if it lands one credible smoke run plus a clear report.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/train`

Stay on a training/execution-oriented branch.
Do not absorb Stage 5.10 main-run scope here.

==================================================
4. REQUIRED INHERITANCE
==================================================

From the Stage 5 docs and prior issues, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Dataset / optics / encoder / loss / trainer path is already frozen enough to run.
3. Regular eval + bicubic baseline already exist from Issue 5.7.
4. Blind line-pair hook already exists from Issue 5.8.
5. Smoke run should consume these paths rather than redefining them.

Important inheritance detail:

- A smoke run may use a reduced budget, but it must be explicitly labeled as smoke.
- Any reduced subset size / step count / epoch budget must be logged as a smoke-only choice.
- If you need a dedicated smoke config, make it explicit rather than overloading
  the short trainer sanity config invisibly.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the dedicated Stage 5 smoke-run config and command
2. the Stage 5 smoke output directory naming
3. the Stage 5 smoke report structure
4. the minimal set of eval evidence attached to the smoke report

This issue MUST keep the following explicit and auditable:

1. smoke run budget
2. depth / config choice used for smoke
3. whether resume was used
4. loss trend and stability observations
5. where checkpoints / previews / summaries are stored
6. whether regular eval and blind eval were invoked after the smoke run

This issue MUST NOT finalize:

1. main-run budget
2. Stage 6 experiment plan
3. new trainer semantics

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real smoke-run path needed for Stage 5 acceptance.

Prefer a narrow structure such as:

- one dedicated smoke config under `configs/stage5/`
- one smoke output directory under `outputs/stage5/`
- one concise execution report:
  `docs/execution/stage5_smoke_report.md`

The implementation should likely include:

- a Stage 5 smoke training command using the existing trainer
- explicit artifact collection from the run
- at least one post-run regular eval call using the existing `5.7` path
- optional blind-eval invocation using the existing `5.8` path if it is cheap enough
- a report summarizing:
  - config / command
  - whether training was stable
  - key loss numbers
  - artifact paths
  - whether the smoke run is sufficient to unblock 5.10

If existing scripts already support the needed behavior, prefer adding config and
reporting over editing code.

==================================================
7. SMOKE-RUN BUDGET REQUIREMENT
==================================================

This issue is specifically about smoke validation, not full reproduction.

You must therefore:

1. choose a modest smoke budget
2. record it explicitly as smoke-only
3. avoid presenting that budget as the final paper-aligned main budget

If compute/time constraints force a very small smoke run, that is acceptable
provided the report clearly says:

- what was actually run
- what remains for the main run
- whether the observed behavior is stable enough to proceed

Do NOT hide resource limits inside ambiguous wording.

==================================================
8. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify the smoke run is real and inspectable.

At minimum, provide:

- smoke training outputs under `outputs/stage5/`
- a config snapshot
- run summary / history / checkpoints / previews
- `docs/execution/stage5_smoke_report.md`

The report should include at least:

- smoke objective and scope boundary
- exact command(s) used
- config path(s)
- artifact path(s)
- key stability observations
- loss-trend summary
- any regular-eval / blind-eval follow-up that was run
- what remains for Issue 5.10

==================================================
9. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- a smoke config under `configs/stage5/`
- `docs/execution/stage5_smoke_report.md`
- possibly `README.md` if a new user-facing smoke command should be documented

Supporting edits are allowed in:

- existing Stage 5 trainer/eval configs

but only if the edits are narrow and clearly smoke-related.

Avoid:

- broad code refactors
- regular eval rewrites
- blind-eval rewrites
- main-run report files

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 5.9 is complete only if:

1. A real paper-aligned smoke run is executed and logged
2. Training completes without NaN / Inf
3. Loss shows at least modestly interpretable behavior
4. Artifacts are complete and traceable
5. A concise smoke report is written
6. The report clearly distinguishes smoke validation from the later main run

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was executed
- exact files changed
- the smoke config / command used
- what artifacts were produced
- the key stability findings
- whether any regular eval / blind eval was run after smoke
- what remains for Issue 5.10
