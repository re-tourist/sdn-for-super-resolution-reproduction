# PROJECT_CONTEXT

## 1. Project Overview

**Project Name:**  
Optical Neural Network Super-Resolution Reproduction

**Primary Goal:**  
Reproduce the paper *Super-resolution image display using diffractive decoders* with a traceable engineering workflow.

**Current Stage:**  
Stage 4 learnability gate has **passed**. The project is now ready to enter **Stage 5: paper-alignment** (GO decision pending implementation).

**Important Stage-4 Principle:**  
The goal is **not** to chase the paper's final metric yet.  
The goal is to answer one question:

> Can the full system learn under gradient-driven training once the encoder is connected to the optical decoder?

---

## 2. Stage Status Snapshot

### Completed / trusted

- Stage 0 repo scaffold and planning docs
- Stage 1 / 2 data sanity path on EMNIST
- interpolation baseline
- pure electronic bottleneck baseline
- unified PSNR / SSIM evaluation helpers
- run-order documentation
- experiment logging / troubleshooting / results summary
- Stage 3 optical decoder skeleton
- Stage 3 optical forward sanity checks
- Stage 3 decoder-only single-sample fitting
- Stage 3 decoder-only small-subset capacity checks
- Stage 4 minimal encoder + hybrid wrapper + dataset path
- Stage 4 minimal trainer with artifact saving
- Stage 4 single-sample closed-loop acceptance: PASS
  - summarized in `docs/execution/stage4_single_sample_report.md`
- Stage 4 small-subset closed-loop acceptance: PASS (with caveats)

### Current focus

- keep Stage 4 evidence as the regression baseline
- use Stage 5 planning docs as the main paper-alignment entry point
- avoid sending large, lagging execution docs to helper agents unless a specific claim needs tracing

---

## 3. What Stage 4 Means In This Repo

Stage 4 in this repo means:

1. connect a minimal encoder
2. feed encoder output into the existing optical contract
3. train on a very small dataset first
4. run single-sample overfit and small-subset sanity checks
5. save enough intermediate artifacts to debug gradients, phase output, and optical readout

Stage 4 does **not** mean:

- full paper hyperparameter alignment
- full `96 -> learned phase pattern -> paper-exact optics` reproduction
- quantization, misalignment, or robustness studies
- large-scale sweeps or ablations

Those belong to later stages.

---

## 4. Hard Constraints A New Assistant Must Know

### 4.1 Optical contract is already frozen

The Stage 3 optical path is already defined as:

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`

Any Stage 5 planning or implementation work must build on that contract rather than redesigning the optical core.

### 4.2 The optical core already exposes the integration hooks we need

The current optical decoder already supports:

- `forward_from_phase(...)`
- `forward_from_field(...)`
- `forward_from_phase_provider(...)`

This means Stage 5 should primarily solve the **paper-aligned encoder / config / loss / eval** problem, not rewrite optics first.

### 4.3 Electronic baseline is already trusted

The pure electronic bottleneck baseline has passed:

- single-sample overfit sanity
- small-subset training sanity

So Stage 4 should reuse it as a training/reference anchor instead of re-opening Stage 1 / 2 questions unless new evidence appears.

### 4.4 Current Stage 3 training scripts are still toy verification scripts

The Stage 3 decoder-only scripts currently use:

- synthetic target patterns
- toy grid sizes such as `24 / 32 / 48 / 20`

They are useful for optical learnability verification, but they are **not** the Stage 5 paper-aligned pipeline.

### 4.5 Some infrastructure is still missing

The repo does **not yet** have:

- a general Stage 4 eval runner
- Stage 5 paper-aligned configs
- a `tests/` directory for systematic automated checks

At the time of this update:

- `scripts/train.py` is empty
- `scripts/eval.py` is empty
- `scripts/visualize.py` is empty

Stage 4 has a **minimal** trainer + dataset path for acceptance runs, but it is not a full paper-aligned training/eval framework.

---

## 5. Current Code Assets That Matter Most

### Optical path

- `src/models/optics/diffractive_decoder.py`
- `src/models/optics/phase_provider.py`
- `src/models/optics/propagation.py`
- `src/models/optics/readout.py`
- `src/models/optics/phase_utils.py`

### Stage 3 verification scripts

- `scripts/check_optical_readout.py`
- `scripts/check_optical_forward_depths.py`
- `scripts/train_decoder_only_single_sample.py`
- `scripts/train_decoder_only_small_subset.py`
- `scripts/train_decoder_only_small_subset_sweep.py`

### Stage 1 / 2 trusted baseline path

- `scripts/train_electronic_baseline.py`
- `src/models/electronic_baseline.py`
- `src/eval/evaluator.py`
- `src/eval/metrics.py`

---

## 6. Stage-5 Review Facts A Helper Agent Should Assume

If a helper agent is asked to review or generate prompts for Stage 5, it should assume the following unless newer code disproves it:

1. Stage 4 learnability gate has passed, but quality is still far from paper-final.
2. Stage 5 is the first phase that should align to the paper's full settings.
3. The optical core contract remains:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
4. Stage 5 must not silently pull Stage 6 scope forward:
   - no quantization sweep
   - no robustness / misalignment study
   - no large ablation matrix
5. The canonical Stage 4 acceptance evidence lives in:
   - `docs/execution/stage4_single_sample_report.md`
   - `docs/execution/stage4_small_subset_report.md`

---

## 7. Current Documentation Sources Of Truth

For Stage 5 review or prompt-generation tasks, prioritize documents in this order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/paper/paper_notes.md`
4. `docs/plan/stage_plan/stage5/stage5_plan.md`
5. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
6. current code in `src/` and `scripts/`

Only pull in the following if a specific claim must be verified:

- `docs/plan/plan_overview.md`
- `docs/plan/stage_plan/stage4/stage4_protocol_freeze.md`
- `docs/execution/stage4_single_sample_report.md`
- `docs/execution/stage4_small_subset_report.md`
- `docs/execution/results_summary.md`
- `docs/execution/experiment_log.md`

This ordering is intentional: the execution logs are useful, but they are verbose, partially redundant, and more likely to lag than the curated context + Stage 5 planning docs.

---

## 8. Immediate Stage-5 Engineering Gaps

The most likely Stage 5 work items are:

1. align optics/encoder configs with paper settings
2. scale data protocol beyond minimal EMNIST setup
3. define Stage 5 training schedule and evaluation pipeline
4. keep Stage 4 artifacts as traceable baseline for regressions

---

## 9. Known Risks

The main current risks are:

1. confusing Stage 3 toy verification with Stage 4 real closed-loop training
2. jumping straight to paper-aligned full settings before proving end-to-end learnability
3. hiding unresolved choices such as phase range mapping or training scale selection
4. writing a large trainer before defining a small acceptance protocol
5. losing traceability by not saving configs, summaries, and intermediate visualizations

---

## 10. Recommended Minimal File Package For Stage-5 Review / Prompting

If a user wants another assistant to review Stage 5 plans or generate prompts for Codex, the default minimal package should be only these 5 files:

- `AGENTS.md`
- `docs/ai/PROJECT_CONTEXT.md`
- `docs/paper/paper_notes.md`
- `docs/plan/stage_plan/stage5/stage5_plan.md`
- `docs/plan/stage_plan/stage5/stage5_issue_plan.md`

Add these only when the task explicitly needs them:

- `docs/plan/plan_overview.md`
- `docs/plan/stage_plan/stage4/stage4_protocol_freeze.md`
- `docs/execution/stage4_single_sample_report.md`
- `docs/execution/stage4_small_subset_report.md`
- `docs/execution/experiment_log.md`
- `docs/execution/results_summary.md`

Do **not** send large execution logs by default just because they exist. Send them only when the helper agent needs to verify a factual claim, a run result, or a documentation gap.

---

## 11. One-Screen Summary

This is a long-running reproduction project for a hybrid
`encoder -> diffractive optical decoder` super-resolution paper.

The repo has already passed:

- Stage 1 / 2 baseline and evaluation groundwork
- Stage 3 optical module verification
- Stage 4 minimal closed-loop learnability acceptance

The repo is now preparing Stage 5:

- align the system to the paper's data / optics / loss / eval settings
- keep Stage 4 as the trusted learnability baseline
- avoid drifting into Stage 6 ablations too early

For prompt-writing or review tasks, the most efficient context package is:
`AGENTS.md` + `PROJECT_CONTEXT.md` + `paper_notes.md` + `stage5_plan.md` + `stage5_issue_plan.md`.
