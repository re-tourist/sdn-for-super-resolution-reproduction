You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.7:

Issue title:
Add Stage 5 evaluation runner with bicubic baseline

This is a Stage-5-scoped regular-eval task.
Do not turn it into a blind-test task, a trainer refactor, or a Stage 6 study.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/paper/paper_notes.md`
4. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
5. `docs/plan/stage_plan/stage5/stage5_plan.md`
6. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
7. `configs/stage5/stage5_emnist_display.yaml`
8. `configs/stage5/stage5_optics_paper_aligned.yaml`
9. `configs/stage5/stage5_paper_encoder.yaml`
10. `configs/stage5/stage5_sr_loss.yaml`
11. `configs/stage5/stage5_trainer_short.yaml`

Then inspect these implementation-reference files before editing:

12. `scripts/train_stage5_paper.py`
13. `src/eval/metrics.py`
14. `src/eval/evaluator.py`
15. `src/datasets/stage5_emnist_display.py`
16. `src/datasets/target_adapter.py`

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- For unresolved items, obey the ownership rules in `stage5_protocol_freeze.md`.
- This issue owns regular eval + bicubic baseline only.
- Blind line-pair / resolution-target evaluation belongs to Issue 5.8, not here.

==================================================
1. ISSUE 5.7 GOAL
==================================================

Implement a Stage 5 regular evaluation path that can:

1. load a Stage 5 paper-path checkpoint
2. run reproducible val/test evaluation
3. compute PSNR / SSIM
4. compute and report a bicubic baseline
5. save auditable eval artifacts and assumptions

The result should provide a narrow, usable Stage 5 eval runner for later smoke
and main-run reporting.

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not add blind line-pair generation or reporting
- do not redesign the Stage 5 trainer
- do not reopen dataset tiling rules or split semantics
- do not reopen optics distance mapping or crop/FOV semantics
- do not reopen encoder phase range or loss policy
- do not build a broad repo-wide evaluation framework
- do not add Stage 6 sweeps, robustness, quantization, or ablation logic
- do not silently change normalization or crop assumptions inside metric code

This issue is successful only if it lands a narrow Stage-5 regular eval path
with explicit bicubic-baseline behavior.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/eval`

Stay on an evaluation-oriented branch.
Do not use this issue to absorb trainer or blind-eval scope.

==================================================
4. REQUIRED INHERITANCE
==================================================

From `stage5_protocol_freeze.md`, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Stage 4 artifacts are regression baseline only.
3. Issue 5.3 already froze crop / FOV alignment semantics.
4. Issue 5.4 already froze the Stage 5 encoder-facing `phi_lr` size and phase path.
5. Issue 5.5 already froze the loss-side Stage 5 defaults.
6. Issue 5.6 already landed the Stage 5 trainer / checkpoint path.
7. Issue 5.7 must consume those paths, not silently redefine them.

Important inheritance detail:

- Regular eval must compare predictions against the frozen ROI-aligned target path.
- Crop size, crop origin, and normalization assumptions must be logged, not hidden.
- Metric helpers already exist in `src/eval/metrics.py` and `src/eval/evaluator.py`.
  Reuse them unless a very small extension is strictly needed.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the Stage 5 regular-eval script path and CLI/config interface
2. the eval artifact directory structure under `outputs/stage5/`
3. the exact reporting schema for PSNR / SSIM outputs
4. the bicubic-baseline implementation details needed for reproducibility
5. the explicit recorded assumptions for split / crop / normalization / baseline scale

This issue MUST keep the following explicit and auditable:

1. which checkpoint is evaluated
2. which split is evaluated
3. what tensor is scored against what target tensor
4. how the bicubic baseline is generated
5. whether anti-aliasing is enabled
6. what crop / normalization assumptions were used

This issue MUST NOT finalize:

1. blind line-pair protocol
2. new crop semantics
3. new dataset split rules
4. new trainer architecture

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real Stage 5 regular-eval path needed for smoke and
main-run reporting.

Prefer a narrow structure such as:

- `scripts/eval_stage5_paper.py`
- a Stage 5 eval config under `configs/stage5/` if helpful
- small reusable helpers under `src/eval/` only if strictly needed

The implementation should likely include:

- checkpoint loading for the Stage 5 encoder + diffractive decoder path
- Stage 5 dataset loading for `val` and/or `test`
- ROI-aligned target generation using the frozen target-adapter semantics
- metric computation using the existing PSNR / SSIM helpers
- bicubic baseline generation with anti-aliasing
- saved eval summary artifacts

Do not broaden the repo's empty global eval entrypoints just because they exist.
Keep this Stage-5-specific and practical.

==================================================
7. BICUBIC BASELINE REQUIREMENT
==================================================

The paper notes and Stage 5 docs require a bicubic baseline with anti-aliasing.

You must make the baseline path explicit and reproducible.

At minimum, record:

1. the low-resolution spatial size or scale factor used by the baseline
2. whether the baseline is implemented as downsample -> upsample
3. where anti-aliasing is enabled
4. the interpolation mode used in each step

If the paper/docs do not justify one uniquely final baseline degradation path,
do NOT fake certainty.

Instead:

- make the baseline resolution / scale configurable
- pick a clear Stage 5 default
- log that default in config and summary

Do NOT:

- use nearest-neighbor as the baseline
- hide baseline resize assumptions in code without logging
- silently choose a baseline crop that disagrees with the frozen ROI semantics

==================================================
8. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify that the eval path is real and inspectable.

At minimum, provide:

- a Stage 5 eval script
- any small Stage 5 eval config needed
- an eval output directory under `outputs/stage5/`
- a saved summary containing:
  - checkpoint path
  - evaluated split
  - PSNR / SSIM for the model path
  - PSNR / SSIM for the bicubic baseline
  - crop / normalization assumptions
  - bicubic baseline assumptions

Prefer also saving a small preview artifact that makes comparison easy, such as:

- input
- target
- model prediction
- bicubic baseline
- absolute-error views

If the repo already has a good Stage 5 artifact style, reuse it.

==================================================
9. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- `scripts/eval_stage5_paper.py`
- a Stage 5 eval config under `configs/stage5/`
- narrow helper additions under `src/eval/` only if strictly needed

Supporting edits are allowed in:

- `README.md`

but only if you add a new user-facing eval command that should be documented.

Avoid:

- trainer refactors
- optics-core changes
- dataset-protocol rewrites
- blind-test code

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 5.7 is complete only if:

1. A real Stage 5 regular eval path exists
2. PSNR / SSIM can be computed on a Stage 5 split
3. A bicubic baseline is computed reproducibly with anti-aliasing
4. Metric outputs and evaluation assumptions are saved and inspectable
5. The implementation stays separate from blind line-pair concerns
6. No frozen Stage 5 crop / normalization semantics are silently changed

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was implemented
- exact files changed
- how the Stage 5 regular eval path works
- how the bicubic baseline is generated
- what command(s) were executed for sanity
- what artifacts were produced
- what was intentionally left for Issue 5.8
