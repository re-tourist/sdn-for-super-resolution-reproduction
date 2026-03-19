# Results Summary

This document summarizes the current repo-level execution verdicts.
It is intentionally evidence-first and separates engineering completion
from paper-quality success claims.

## Current Verdict

- Stage 5 engineering completion: `PASS`
- Paper-quality reproduction claim: `NOT YET DEMONSTRATED`
- Stage 6 entry decision: `GO`

Why this is the current verdict:

- The full Stage 5 paper-aligned pipeline now exists end to end:
  dataset, optics configs, phase-only encoder, Stage 5 loss, trainer,
  regular eval, and blind eval hook.
- A real Stage 5 smoke run exists and completed stably.
- A real full `phase-only / L=5` main run exists.
- Matching regular eval and blind eval artifacts exist for that main run.
- The reported model PSNR/SSIM results do not currently beat the bicubic
  baseline, so Stage 5 is an engineering completion milestone rather
  than a paper-final quality success milestone.

## Primary Stage 5 Evidence

- `docs/execution/stage5_smoke_report.md`
- `docs/execution/stage5_main_run_report.md`
- `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`

## What Stage 5 Demonstrated

### 1. The paper-aligned pipeline is real and reproducible

Stage 5 landed the following paper-path components in the repo:

- EMNIST `96x96` display dataset path
- Stage 5 optics configs for `L=1/3/5`
- phase-only encoder with explicit `phi_lr` semantics
- Stage 5 super-resolution loss with configurable efficiency term
- Stage 5 trainer with checkpointing and artifact saving
- regular val/test eval with bicubic baseline
- blind line-pair artifact path

This means the repo now has a credible paper-aligned starting point for
systematic follow-up work.

### 2. Smoke training showed the pipeline is stable enough to launch

The smoke run recorded in `docs/execution/stage5_smoke_report.md` showed:

- end-to-end training completed without NaN / Inf
- checkpoints, previews, and summaries were written successfully
- regular eval and blind eval hooks both ran after training

That smoke result is not a quality claim, but it is valid engineering
evidence that the Stage 5 stack is launchable and inspectable.

### 3. A full `phase-only / L=5` main run completed

From `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`:

- run name: `stage5_l5_phase_main`
- depth: `L=5`
- completed steps: `750000`
- equivalent budget: `500` epochs
- best step: `745500`
- best val loss: `0.0188692901`
- final train loss: `0.0201732814`
- final val loss: `0.0188748985`
- history storage mode: `jsonl_incremental`

This is real long-run execution evidence, not just launch preparation.

### 4. Matching regular eval artifacts exist for the same best checkpoint

The regular eval summaries score `I_out_roi` against the frozen
`target_roi` path and use the Stage 5 bicubic baseline with
`32x32 -> 96x96` downsample-then-upsample.

| split | model PSNR | model SSIM | bicubic PSNR | bicubic SSIM |
| --- | ---: | ---: | ---: | ---: |
| `val` | `14.9181` | `0.8126` | `25.6875` | `0.9448` |
| `test` | `9.5774` | `0.4386` | `20.2487` | `0.8393` |

Artifact sources:

- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`

### 5. Matching blind eval artifacts exist for the same best checkpoint

From `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`:

- blind target family: `line_pair`
- deterministic blind target count: `24`
- target canvas: `96x96`
- target orientations: `horizontal`, `vertical`
- line widths: `1 px`, `2 px`
- gap ladder: `1, 2, 3, 4, 6, 8 px`

This demonstrates that the blind-test hook is real, deterministic, and
artifact-producing for the Stage 5 main checkpoint.

## What Stage 5 Did Not Demonstrate

The current evidence does **not** justify the following stronger claims:

- It does not show paper-final quality reproduction.
- It does not show that the current model is competitive with the
  bicubic baseline on the reported PSNR / SSIM summaries.
- It does not show a strong blind-resolution success claim.
- It does not complete Stage 6 studies such as quantization,
  robustness, misalignment, or systematic ablations.

The most important empirical caveat is explicit:

- On both `val` and `test`, the reported bicubic baseline outperforms
  the current Stage 5 model on PSNR and SSIM.

The blind-eval caveat is also explicit:

- The Stage 5 blind result is artifact-first.
- It saves reproducible target and output grids.
- It does not introduce or justify a paper-final scalar blind score.

## What Still Matters From Stage 4

Stage 4 remains the trusted learnability baseline.
Stage 5 should be read as a paper-aligned engineering extension on top
of Stage 4, not as a replacement for that earlier acceptance evidence.

This distinction matters because:

- Stage 4 answered whether the encoder + optics path can learn at all.
- Stage 5 answered whether the repo can run the full paper-aligned path
  end to end with real artifacts.
- Stage 6 should now investigate why paper-quality performance has not
  yet been demonstrated.

## Stage 6 Entry Decision

Decision: `GO`

Reasoning:

- The repo now has a real, reproducible, paper-aligned starting point.
- The main run, regular eval, and blind eval all exist as auditable
  artifacts.
- The remaining problem is no longer "can we build or run the Stage 5
  path?" but rather "why does the current Stage 5 path underperform the
  bicubic baseline, and which Stage 6 experiments are needed to explain
  that gap?"

Stage 6 should therefore begin from the current frozen Stage 5 pipeline,
not by re-litigating whether the Stage 5 engineering path exists.
