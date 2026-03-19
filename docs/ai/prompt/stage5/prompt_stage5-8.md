You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.8:

Issue title:
Add Stage 5 blind line-pair generator and evaluation hook

This is a Stage-5-scoped blind-eval task.
Do not turn it into a rewrite of the regular eval runner, a trainer task, or a
Stage 6 robustness study.

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
7. `configs/stage5/stage5_optics_paper_aligned.yaml`
8. `configs/stage5/stage5_paper_encoder.yaml`
9. `configs/stage5/stage5_eval.yaml`

Then inspect these implementation-reference files before editing:

10. `scripts/eval_stage5_paper.py`
11. `scripts/train_stage5_paper.py`
12. `src/eval/metrics.py`
13. `src/eval/evaluator.py`
14. `src/datasets/target_adapter.py`

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- For unresolved items, obey the ownership rules in `stage5_protocol_freeze.md`.
- This issue owns blind line-pair / resolution-target generation plus a narrow
  eval hook only.
- Regular PSNR / SSIM eval and bicubic baseline were already handled by Issue 5.7.

==================================================
1. ISSUE 5.8 GOAL
==================================================

Implement a Stage 5 blind line-pair / resolution-target path that can:

1. generate deterministic blind-test targets not used in the training dataset
2. run the current Stage 5 model path on those targets
3. save auditable artifacts for visual inspection and later reporting
4. plug into the Stage 5 eval workflow without rewriting the regular eval path

The result should provide a narrow blind-test capability that complements, but
does not replace, the regular eval runner from Issue 5.7.

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not redesign `scripts/eval_stage5_paper.py` into a broad benchmark framework
- do not reopen PSNR / SSIM or bicubic-baseline design from Issue 5.7
- do not change Stage 5 dataset split rules or EMNIST protocol
- do not add quantization, misalignment, robustness, or ablation studies
- do not silently change crop / normalization semantics
- do not invent a heavy paper-final scalar metric if the documents do not freeze one
- do not mix this with smoke-run or main-run orchestration

This issue is successful only if it lands a narrow, reproducible blind-test path
with clear target-generation assumptions and saved artifacts.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/eval`

Stay on an evaluation-oriented branch.
Do not absorb trainer or Stage 6 scope here.

==================================================
4. REQUIRED INHERITANCE
==================================================

From `stage5_protocol_freeze.md`, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Stage 5 regular eval already exists and must remain the base path.
3. Crop / FOV alignment semantics are already frozen by Issue 5.3.
4. Encoder-facing `phi_lr` semantics are already frozen by Issue 5.4.
5. Trainer / checkpoint loading path is already frozen by Issue 5.6.
6. Blind eval must reuse those paths rather than silently redefining them.

Important inheritance detail:

- Blind targets are separate from the EMNIST training/eval dataset.
- The blind-test hook should reuse the existing Stage 5 checkpoint-loading and
  forward path where practical.
- If a scalar blind-test score is not well specified by the docs, do not fake
  certainty. Prefer reproducible artifacts + explicit metadata.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the Stage 5 blind-target generator path and config interface
2. the artifact directory structure for blind-test outputs
3. the reporting schema for blind-test metadata
4. the narrow hook connecting blind targets to the existing Stage 5 eval/model path
5. the default Stage 5 blind-target parameter set, if kept configurable and logged

This issue MUST keep the following explicit and auditable:

1. target canvas size
2. line width / spacing / orientation assumptions
3. whether targets are binary or smoothed before display
4. how targets are fed into the Stage 5 model path
5. what outputs are saved and how they are organized

This issue MUST NOT finalize:

1. a repo-wide eval framework
2. new regular-eval semantics
3. new crop semantics
4. Stage 6 experiment scope

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real blind-test path needed for Stage 5.

Prefer a narrow structure such as:

- a blind-target utility under `src/eval/` or another narrow, local module
- a Stage 5 blind-eval config under `configs/stage5/`
- either:
  - a small dedicated script such as `scripts/eval_stage5_blind_linepair.py`, or
  - a very small additive hook to `scripts/eval_stage5_paper.py`

Prefer separation over entanglement.
If a separate script is cleaner, use a separate script.

The implementation should likely include:

- deterministic line-pair / resolution-target generation
- explicit config for target geometry and target count
- checkpoint loading and model forward reuse
- saved visual artifacts for:
  - generated blind target
  - model output
  - optional comparison view / profile view if narrowly justified
- a summary file with target metadata and output paths

==================================================
7. BLIND-TARGET GENERATION REQUIREMENT
==================================================

The paper notes mention blind testing with line pairs / resolution targets, but
the exact engineering form may not be fully frozen.

You must therefore:

1. make target generation deterministic
2. make key geometry parameters configurable
3. pick a clear Stage 5 default set
4. log those defaults in config and summary

If the docs do not uniquely settle details such as:

- exact line widths
- exact spacing ladder
- exact orientation set
- exact number of targets

then do NOT present one choice as paper-unique truth.

Instead:

- implement a reasonable Stage 5 default
- keep it configurable
- document it as a Stage 5 engineering choice

Do NOT:

- generate targets with hidden randomness
- bury target geometry only in code
- make blind targets depend on the EMNIST dataset path

==================================================
8. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify that the blind-test path is real and inspectable.

At minimum, provide:

- the blind-target generator utility or module
- a Stage 5 blind-eval config if needed
- a blind-test output directory under `outputs/stage5/`
- a saved summary containing:
  - checkpoint path
  - blind-target generation config / metadata
  - output artifact paths
  - any narrow reporting fields you define

Prefer also saving preview artifacts such as:

- a grid of generated blind targets
- the corresponding model outputs
- optional line-profile or side-by-side comparison figures if that is small and clear

If you add any scalar reporting, keep it narrow and explain what it means.
Do not over-claim it as the paper's final blind metric unless the docs clearly support that.

==================================================
9. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- a new blind-eval script under `scripts/`
- a Stage 5 blind-eval config under `configs/stage5/`
- a narrow helper module under `src/eval/` or another small local path

Supporting edits are allowed in:

- `README.md`
- `scripts/eval_stage5_paper.py`

but only for a small, well-scoped hook or command reference.

Avoid:

- trainer refactors
- optics-core edits
- dataset rewrites
- broad metric-framework changes

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 5.8 is complete only if:

1. Blind line-pair / resolution targets can be generated deterministically
2. The current Stage 5 model path can be executed on those targets
3. Blind-test artifacts are saved in a reproducible format
4. Target-generation assumptions are explicit and logged
5. The implementation stays separate from regular eval concerns
6. No frozen Stage 5 crop / normalization semantics are silently changed

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was implemented
- exact files changed
- how blind targets are generated
- how the blind-test hook reuses the Stage 5 model/eval path
- what command(s) were executed for sanity
- what artifacts were produced
- what was intentionally left unresolved for later issues
