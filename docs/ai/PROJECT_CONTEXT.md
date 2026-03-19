# PROJECT_CONTEXT

## 1. Project Overview

**Project Name:**  
Optical Neural Network Super-Resolution Reproduction

**Primary Goal:**  
Reproduce the paper *Super-resolution image display using diffractive decoders*
with a traceable engineering workflow.

**Current Stage:**  
Stage 5 paper-aligned engineering completion has **passed**.
The repo is now ready to enter **Stage 6** for systematic follow-up
experiments.

**Current high-level claim:**  
The repo now has a real paper-aligned pipeline and real execution
artifacts, but it does **not** yet have evidence for paper-final quality
reproduction.

---

## 2. Current Repo Verdict

Three separate statements must stay separate:

1. Stage 5 engineering completion: `PASS`
2. Paper-quality reproduction claim: `NOT YET DEMONSTRATED`
3. Stage 6 entry decision: `GO`

This separation matters because:

- the pipeline is real and reproducible
- the main run and matching eval artifacts exist
- but the current model still underperforms the bicubic baseline on the
  reported PSNR / SSIM summaries

---

## 3. Stage Status Snapshot

### Completed / trusted

- Stage 0 repo scaffold and planning docs
- Stage 1 / 2 data sanity path on EMNIST
- interpolation baseline
- pure electronic bottleneck baseline
- unified PSNR / SSIM evaluation helpers
- Stage 3 optical contract and forward sanity
- Stage 3 decoder-only optical capacity checks
- Stage 4 single-sample closed-loop acceptance: PASS
- Stage 4 small-subset closed-loop acceptance: PASS
- Stage 5 protocol freeze
- Stage 5 paper-aligned dataset path
- Stage 5 paper-aligned optics configs
- Stage 5 phase-only encoder
- Stage 5 SR loss
- Stage 5 trainer with artifact saving and resume support
- Stage 5 regular eval with bicubic baseline
- Stage 5 blind line-pair hook
- Stage 5 smoke run
- Stage 5 full `phase-only / L=5` main run
- Stage 5 matching regular eval and blind eval artifacts

### Current focus

- preserve Stage 4 as the learnability baseline
- preserve Stage 5 as the paper-aligned engineering baseline
- use Stage 6 for systematic follow-up experiments rather than redoing
  Stage 5 implementation work

---

## 4. What Stage 5 Means In This Repo

Stage 5 in this repo means:

1. align the system to the paper-path data / optics / encoder / loss /
   trainer / eval settings
2. run a real smoke check
3. run a real full `phase-only / L=5` main training path
4. record matching regular eval and blind eval artifacts
5. preserve enough summaries, configs, and outputs for auditability

Stage 5 does **not** mean:

- paper-final quality has already been reproduced
- the current model has beaten the bicubic baseline
- blind line-pair artifacts have already become a strong scalar success
  claim
- Stage 6 quantization, robustness, misalignment, or ablation work has
  already been performed

---

## 5. Hard Constraints A New Assistant Must Know

### 5.1 Optical contract is still frozen

The optical path remains:

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`

Stage 6 work must build on that contract unless a future task explicitly
reopens it with evidence and scope approval.

### 5.2 Stage 4 remains the trusted learnability baseline

Stage 4 answered the "can the encoder + optics path learn at all?"
question and should not be erased from the repo narrative.

Stage 5 is an engineering extension on top of that baseline, not a
replacement for it.

### 5.3 Stage 5 artifacts now exist and are the current paper-path anchor

The most important Stage 5 execution artifacts are:

- `docs/execution/stage5_smoke_report.md`
- `docs/execution/stage5_main_run_report.md`
- `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`

### 5.4 The key empirical caveat must stay visible

Current Stage 5 evidence shows:

- the pipeline is real
- the run artifacts are real
- but bicubic currently outperforms the model on the reported PSNR /
  SSIM summaries for both `val` and `test`

This caveat must not be hidden in future prompts or summary docs.

### 5.5 The blind eval result is artifact-first

The Stage 5 blind hook is useful and real, but it currently supports:

- deterministic blind target generation
- saved blind target and output artifacts
- human inspection and later follow-up analysis

It does **not** currently justify a stronger paper-final blind metric
claim.

---

## 6. Current Code Assets That Matter Most

### Paper-path model and optics

- `src/models/encoders/paper_phase_encoder.py`
- `src/models/optics/diffractive_decoder.py`
- `src/models/optics/phase_provider.py`
- `src/models/optics/readout.py`

### Stage 5 data / loss / eval

- `src/datasets/stage5_emnist_display.py`
- `src/datasets/target_adapter.py`
- `src/losses/stage5_sr_loss.py`
- `src/eval/metrics.py`
- `src/eval/evaluator.py`
- `src/eval/stage5_blind_targets.py`

### Stage 5 execution scripts

- `scripts/train_stage5_paper.py`
- `scripts/eval_stage5_paper.py`
- `scripts/eval_stage5_blind_linepair.py`

### Stage 5 configs

- `configs/stage5/stage5_emnist_display.yaml`
- `configs/stage5/stage5_optics_paper_aligned.yaml`
- `configs/stage5/stage5_paper_encoder.yaml`
- `configs/stage5/stage5_sr_loss.yaml`
- `configs/stage5/stage5_trainer_smoke.yaml`
- `configs/stage5/stage5_trainer_main.yaml`
- `configs/stage5/stage5_eval.yaml`
- `configs/stage5/stage5_blind_eval.yaml`

---

## 7. What A Helper Assistant Should Assume Now

If a helper agent is asked to review or extend the repo after Stage 5,
it should assume the following unless newer code disproves it:

1. Stage 5 engineering completion has passed.
2. Stage 5 empirical quality is still below the bicubic baseline on the
   reported PSNR / SSIM summaries.
3. The repo is ready to enter Stage 6 because the paper-aligned pipeline
   and evidence base are real.
4. Stage 4 remains the trusted learnability baseline for regressions.
5. Stage 6 should not rebuild Stage 5 from scratch unless a concrete
   blocker or regression is found.

---

## 8. Current Documentation Sources Of Truth

For Stage 6 review or prompt-generation tasks, prioritize documents in
this order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
4. `docs/execution/stage5_main_run_report.md`
5. `docs/execution/results_summary.md`
6. `docs/execution/experiment_log.md`
7. `docs/paper/paper_notes.md`

Add these when the task needs historical scope or issue ownership:

- `docs/plan/stage_plan/stage5/stage5_plan.md`
- `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
- `docs/execution/stage5_smoke_report.md`

Only pull in older Stage 3 / Stage 4 execution logs when a specific
claim must be verified.

---

## 9. One-Screen Summary

This is a long-running reproduction project for a hybrid
`encoder -> diffractive optical decoder` super-resolution paper.

The repo has already passed:

- Stage 3 optical module verification
- Stage 4 minimal closed-loop learnability acceptance
- Stage 5 paper-aligned engineering completion

The repo has **not** yet shown:

- paper-final quality reproduction
- superiority over the bicubic baseline on the reported regular-eval
  summaries

The current next-step decision is:

- enter Stage 6 with a `GO` decision
- keep Stage 4 and Stage 5 artifacts as regression anchors
- use Stage 6 to investigate the quality gap rather than questioning
  whether the Stage 5 pipeline exists
