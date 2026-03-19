# Experiment Log

This document records the major execution milestones that have actually
been implemented or run in this repo.

It is chronological and evidence-first.
Planned future work belongs in planning docs, not here.

## Stage 3: Optical Module Verification

### S3 optical contract and readout path landed

Key outcome:

- The optical contract was frozen as
  `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`.
- Full-grid intensity and ROI readout were separated cleanly.
- Crop behavior became explicit and auditable.

Key references:

- `src/models/optics/diffractive_decoder.py`
- `src/models/optics/readout.py`
- `scripts/check_optical_readout.py`

### S3 forward sanity passed for `L=1/3/5`

Key outcome:

- The current optical core could run forward for `L=1/3/5`
  without changing the contract.

Key reference:

- `scripts/check_optical_forward_depths.py`

### S3 decoder-only capacity checks passed

Key outcome:

- Decoder-only single-sample fitting and small-subset depth checks showed
  that the optical decoder stack has learnable capacity under the Stage 3
  verification protocol.

Key references:

- `outputs/optics/decoder_only_single_sample_smoke/summary.json`
- `outputs/optics/decoder_only_small_subset_sweep_run1/sweep_summary.json`

Important boundary:

- These were optical verification results, not paper-aligned Stage 5
  results.

## Stage 4: Minimal Closed-Loop Learnability Gate

### S4 single-sample acceptance passed

Key outcome:

- End-to-end training with encoder + optics could overfit a single sample
  without NaN / Inf.

Key references:

- `docs/execution/stage4_single_sample_report.md`
- `outputs/stage4/minimal_trainer/issue4_6_single_sample_100/run_summary.json`
- `outputs/stage4/minimal_trainer/issue4_6_single_sample_300/run_summary.json`

### S4 small-subset acceptance passed

Key outcome:

- The minimal closed-loop path trained on a small real subset with
  interpretable loss behavior and observable gradients.

Key references:

- `docs/execution/stage4_small_subset_report.md`
- `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s200/run_summary.json`
- `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s400/run_summary.json`

Stage-level meaning:

- Stage 4 answered the learnability question.
- Stage 4 remains the trusted regression baseline for later stages.

## Stage 5: Paper-Aligned Pipeline Landing

### S5.1 protocol freeze landed

Key outcome:

- The Stage 5 scope, inherited contracts, paper-aligned defaults,
  unresolved-item ownership, and Stage 6 boundary were frozen explicitly.

Key reference:

- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`

### S5.2 through S5.8 landed the full paper-path stack

Key outcome:

- Stage 5 dataset path landed.
- Stage 5 optics configs for `L=1/3/5` landed.
- Stage 5 phase-only encoder landed.
- Stage 5 SR loss landed.
- Stage 5 trainer landed.
- Stage 5 regular eval with bicubic baseline landed.
- Stage 5 blind line-pair hook landed.

Key references:

- `configs/stage5/stage5_emnist_display.yaml`
- `configs/stage5/stage5_optics_paper_aligned.yaml`
- `configs/stage5/stage5_paper_encoder.yaml`
- `configs/stage5/stage5_sr_loss.yaml`
- `configs/stage5/stage5_trainer_short.yaml`
- `configs/stage5/stage5_eval.yaml`
- `configs/stage5/stage5_blind_eval.yaml`

### S5.9 smoke training completed

Key outcome:

- A real paper-aligned smoke run completed.
- The run stayed finite and produced checkpoints, previews, and summaries.
- Post-run regular eval and blind eval both executed successfully.

Key reference:

- `docs/execution/stage5_smoke_report.md`

### S5.10 full `phase-only / L=5` main run completed

Key outcome:

- A full Stage 5 main run completed with the paper-aligned `L=5` path.
- Matching regular eval and blind eval artifacts were generated for the
  same best checkpoint.

Main run facts:

- run name: `stage5_l5_phase_main`
- completed steps: `750000`
- equivalent budget: `500` epochs
- best step: `745500`
- best val loss: `0.0188692901`
- final train loss: `0.0201732814`
- final val loss: `0.0188748985`

Regular eval facts:

| split | model PSNR | model SSIM | bicubic PSNR | bicubic SSIM |
| --- | ---: | ---: | ---: | ---: |
| `val` | `14.9181` | `0.8126` | `25.6875` | `0.9448` |
| `test` | `9.5774` | `0.4386` | `20.2487` | `0.8393` |

Blind eval facts:

- deterministic `line_pair` target family
- `24` blind targets
- artifact-first blind reporting

Key references:

- `docs/execution/stage5_main_run_report.md`
- `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`

### S5.11 consolidation verdict

Key outcome:

- Stage 5 engineering completion verdict: `PASS`
- Paper-quality reproduction claim: `NOT YET DEMONSTRATED`
- Stage 6 entry decision: `GO`

Reason for the caveat:

- On both reported regular-eval splits, bicubic outperforms the current
  model on PSNR and SSIM.
- Blind eval provides real artifacts but does not justify a stronger
  paper-final blind-resolution claim.

## Current Log Boundary

This log now records the repo through Stage 5 completion and Stage 6
entry.

What comes next belongs to Stage 6:

- quantization follow-up
- robustness / misalignment checks
- systematic ablations
- investigation of why the current Stage 5 model underperforms bicubic

Those follow-up experiments should build on the current Stage 5 evidence
base rather than rewriting the Stage 5 pipeline.
