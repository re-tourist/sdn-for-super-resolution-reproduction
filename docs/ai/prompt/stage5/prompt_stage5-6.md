You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.6:

Issue title:
Implement Stage 5 paper-aligned trainer and configs

This is a Stage-5-scoped trainer task.
Do not turn it into a generic training framework, a Stage 6 sweep task, or an eval task.

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

Then inspect these implementation-reference files before editing:

10. `scripts/train_stage4_minimal.py`
11. `src/datasets/target_adapter.py`
12. `src/datasets/stage5_emnist_display.py`
13. `src/models/hybrid/minimal_hybrid_wrapper.py`
14. `src/models/encoders/paper_phase_encoder.py`
15. `src/losses/stage5_sr_loss.py`

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- For unresolved items, obey the ownership rules in `stage5_protocol_freeze.md`.
- `5.2` through `5.5` already froze dataset / optics / encoder / loss defaults.
  This trainer must consume them, not silently re-decide them.

==================================================
1. ISSUE 5.6 GOAL
==================================================

Implement a dedicated Stage 5 paper-aligned trainer that can run the current
paper-path pipeline and save auditable artifacts.

The result should provide a reproducible Stage 5 training path that:

1. builds the Stage 5 dataset, encoder, decoder, and loss together
2. supports separate optimizer parameter groups for encoder and decoder
3. supports resume from checkpoint
4. writes complete Stage 5 artifacts
5. is suitable for short sanity runs now and smoke/main runs later

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not turn this into a repo-wide training framework
- do not fill `scripts/train.py` as a generic entrypoint
- do not redesign the Stage 4 trainer for broad reuse
- do not reopen dataset tiling, optics distances, phase range, gamma policy,
  sigma policy, or crop semantics
- do not build the Stage 5 eval runner here
- do not add bicubic baseline or blind line-pair logic here
- do not add Stage 6 sweeps, robustness, or quantization logic

This issue is successful only if it lands a Stage-5-specific trainer path with
clear configs and auditable artifacts.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/train`

Stay on a training-oriented branch.
Do not use this issue to modify `feat/eval` scope.

==================================================
4. REQUIRED INHERITANCE
==================================================

From `stage5_protocol_freeze.md`, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Stage 4 artifacts are regression baseline only
3. The trainer must consume already frozen Stage 5 configs for:
   - dataset
   - optics
   - encoder
   - loss
4. The trainer may add:
   - resume support
   - artifact saving
   - Stage-5-specific config entrypoints
5. The trainer must NOT finalize:
   - tiling rules
   - distance mapping
   - phase range
   - gamma policy
   - sigma granularity
   - crop semantics

Important inheritance detail:

- `Stage5SuperResolutionLoss` now requires explicit `input_power` when the
  efficiency term is enabled.
- This issue owns the narrow trainer-side wiring of that `input_power` input.
- It does NOT own changing `eta` math or changing the loss contract.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the Stage 5 trainer script path and CLI/config interface
2. the artifact directory structure for Stage 5 training runs
3. the resume/checkpoint protocol
4. the trainer-side source of `input_power` passed into the Stage 5 loss
5. the short-run sanity configuration used to verify the trainer end to end

This issue MUST keep the following explicit and auditable:

1. encoder LR vs decoder LR
2. whether the efficiency term is enabled for the run
3. how `input_power` is sourced for the loss call
4. where checkpoints, summaries, metrics, and previews are saved
5. what is a short-run sanity config versus a later smoke/main config

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real Stage 5 trainer path needed for end-to-end sanity.

Prefer a narrow Stage-5-specific structure such as:

- `scripts/train_stage5_paper.py`
- Stage 5 trainer config(s) under `configs/stage5/`
- reuse of existing repo modules where helpful, without generalizing the repo

The implementation should likely include:

- Stage 5 dataset construction using the Stage 5 display dataset path
- model assembly using the Stage 5 encoder + current diffractive decoder
- ROI target adaptation using the existing target-adapter path or a very small
  Stage-5-specific equivalent if strictly needed
- Stage 5 loss wiring using `Stage5SuperResolutionLoss`
- optimizer parameter groups with separate LR for encoder and decoder
- checkpoint saving and resume loading
- short-run artifact saving

You must keep the script practical for later issues:

- a short run should be able to complete now
- later smoke/main issues should be able to reuse the trainer rather than rewrite it

==================================================
7. INPUT POWER WIRING REQUIREMENT
==================================================

Because `5.5` froze the loss interface but intentionally left trainer-side
`input_power` sourcing to `5.6`, you must make that decision explicit here.

You must:

1. choose a concrete trainer-side source for per-sample `input_power`
2. document that choice in config and/or run summary
3. keep it narrow: consume existing forward-path tensors, do not redesign optics

Do NOT:

- hide `input_power` in a constant global without documentation
- change `eta = 100 * P_o / P_i`
- change the loss-side `output_power_source` silently

==================================================
8. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify that the trainer path is real and inspectable.

At minimum, provide:

- a Stage 5 trainer script
- Stage 5 trainer config entries
- a short-run sanity output directory under `outputs/stage5/`
- saved artifacts including:
  - config snapshot
  - run summary
  - metrics history
  - at least one checkpoint
  - at least one preview artifact

If the repo already has a preferred artifact style from Stage 4, reuse the good
parts without turning this into a generalized framework.

==================================================
9. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- `scripts/train_stage5_paper.py`
- Stage 5 trainer config(s) under `configs/stage5/`
- small supporting integration code only if strictly needed

Supporting edits are allowed in:

- `src/models/hybrid/`
- `src/datasets/`

but only if they are narrow and directly required to make the Stage 5 trainer work.

Avoid:

- broad framework refactors
- eval scripts
- loss contract changes
- optics contract changes

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 5.6 is complete only if:

1. A short Stage 5 paper-path run completes successfully
2. Encoder and decoder use separate optimizer parameter groups
3. Resume works at least for a minimal checkpoint/restart path
4. Expected artifacts are generated and logged
5. The trainer remains Stage-5-specific rather than framework-generic
6. No already-frozen Stage 5 policy is silently redefined here

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was implemented
- exact files changed
- how the trainer assembles dataset / model / loss
- how `input_power` is sourced for the loss call
- what artifacts are produced
- what short-run sanity command was executed
- what was intentionally left unresolved for later issues
