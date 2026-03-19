# Stage 5 Main Run Report

Status:
- full Stage 5 `phase-only / L=5` main training has been executed on a Linux server
- this report records the uploaded long-run training artifacts under `outputs/stage5/main_run/stage5_l5_phase_main`
- companion regular eval and blind eval artifacts for this exact main run were also uploaded and are now included below

## 1. Run Identity

- run name: `stage5_l5_phase_main`
- execution host: `Linux server (host name not recorded in the uploaded artifacts)`
- GPU / device: `cuda (exact GPU model not recorded in the uploaded artifacts)`
- start time: `2026-03-19T07:00:40.685304+00:00`
- end time: `2026-03-19T13:19:59.703498+00:00`
- wall-clock duration: `6h 19m 19s`
- status: `completed`

## 2. Config And Commands

Main config:
- `configs/stage5/stage5_trainer_main.yaml`

Training command actually used:

```bash
python scripts/train_stage5_paper.py \
  --config configs/stage5/stage5_trainer_main.yaml \
  --run-name stage5_l5_phase_main \
  --device cuda
```

Resume command actually used, if any:

```bash
N/A
```

Regular eval command(s) actually used:

```bash
python scripts/eval_stage5_paper.py \
  --config configs/stage5/stage5_eval.yaml \
  --checkpoint outputs/stage5/main_run/stage5_l5_phase_main/checkpoints/checkpoint_best.pt \
  --split val \
  --run-name stage5_l5_phase_main_val_eval \
  --device cuda

python scripts/eval_stage5_paper.py \
  --config configs/stage5/stage5_eval.yaml \
  --checkpoint outputs/stage5/main_run/stage5_l5_phase_main/checkpoints/checkpoint_best.pt \
  --split test \
  --run-name stage5_l5_phase_main_test_eval \
  --device cuda
```

Blind eval command actually used:

```bash
python scripts/eval_stage5_blind_linepair.py \
  --config configs/stage5/stage5_blind_eval.yaml \
  --checkpoint outputs/stage5/main_run/stage5_l5_phase_main/checkpoints/checkpoint_best.pt \
  --run-name stage5_l5_phase_main_blind_eval \
  --device cuda
```

## 3. Output Roots

- training root: `outputs/stage5/main_run/stage5_l5_phase_main`
- val eval root: `outputs/stage5/eval/stage5_l5_phase_main_val_eval`
- test eval root: `outputs/stage5/eval/stage5_l5_phase_main_test_eval`
- blind eval root: `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval`

## 4. Training Summary

- planned main budget: `L=5`, `batch_size=40`, `steps=750000`, `validate_every=1500`
- actual completed steps: `750000`
- equivalent epoch budget: `500` epochs (`steps_per_epoch = 1500`)
- best step: `745500`
- best val loss: `0.0188692901`
- final train loss: `0.0201732814`
- final val loss: `0.0188748985`
- resume used: `no`
- NaN / Inf observed: `no visible NaN / Inf; status.json records completed and no failure artifact was uploaded`

Notes:
- the run reached the full target budget rather than stopping at a reduced smoke budget
- `history_storage_mode = jsonl_incremental`, so the full per-step trace lives in `history.jsonl` / `grad_stats.jsonl`
- final status shows `current_epoch = 499` because epochs are zero-indexed internally; this corresponds to the intended 500-epoch budget
- final logged throughput near completion was about `32.96 steps/s`

## 5. Regular Eval Summary

Val metrics:
- model PSNR: `14.9180516516`
- model SSIM: `0.8126175770`
- bicubic PSNR: `25.6875014105`
- bicubic SSIM: `0.9447924524`

Test metrics:
- model PSNR: `9.5773711626`
- model SSIM: `0.4385546151`
- bicubic PSNR: `20.2487178734`
- bicubic SSIM: `0.8392536711`

Note:
- both eval summaries score `I_out_roi` against the frozen `target_roi`
- both eval summaries use the Stage 5 bicubic baseline path with `32x32` downsample-then-upsample and anti-aliasing on the downsample step

## 6. Blind Eval Summary

- blind target family: `line_pair`
- blind target count: `24`
- key artifact path(s): `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_targets_grid.png`, `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_outputs_grid.png`, `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_comparison_grid.png`
- qualitative observation: `line-pair blind outputs were successfully generated for all 24 deterministic targets, but the recorded prediction sums remain far below the binary target sums, so qualitative interpretation should rely on the saved grids rather than claiming a strong blind-resolution result.`

Note:
- the blind hook remains artifact-first and does not claim a paper-final scalar blind metric

## 7. Key Artifacts To Keep

- `outputs/stage5/main_run/stage5_l5_phase_main/train.log`
- `outputs/stage5/main_run/stage5_l5_phase_main/status.json`
- `outputs/stage5/main_run/stage5_l5_phase_main/config_snapshot.json`
- `outputs/stage5/main_run/stage5_l5_phase_main/history.jsonl`
- `outputs/stage5/main_run/stage5_l5_phase_main/history.json`
- `outputs/stage5/main_run/stage5_l5_phase_main/grad_stats.jsonl`
- `outputs/stage5/main_run/stage5_l5_phase_main/grad_stats.json`
- `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`
- `outputs/stage5/main_run/stage5_l5_phase_main/loss_curve.png`
- `outputs/stage5/main_run/stage5_l5_phase_main/loss_curve_points.json`
- `outputs/stage5/main_run/stage5_l5_phase_main/preview_step0.png`
- `outputs/stage5/main_run/stage5_l5_phase_main/preview_best.png`
- `outputs/stage5/main_run/stage5_l5_phase_main/preview_final.png`
- `outputs/stage5/main_run/stage5_l5_phase_main/phi_preview_step0.png`
- `outputs/stage5/main_run/stage5_l5_phase_main/phi_preview_best.png`
- `outputs/stage5/main_run/stage5_l5_phase_main/phi_preview_final.png`
- `outputs/stage5/main_run/stage5_l5_phase_main/checkpoints/checkpoint_best.pt`
- `outputs/stage5/main_run/stage5_l5_phase_main/checkpoints/checkpoint_latest.pt`
- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/preview_comparison.png`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/preview_comparison.png`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_targets_grid.png`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_outputs_grid.png`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_comparison_grid.png`

Notes:
- for this long main run, `history.jsonl` and `grad_stats.jsonl` are the authoritative full per-step traces
- `history.json` and `grad_stats.json` are compact index artifacts that point back to the JSONL traces

## 8. Conclusion

- does the main run complete the Stage 5 launch path with real long-run evidence? `Yes: the full 750000-step L=5 run completed, and matching regular eval plus blind eval artifacts are now present for the same best checkpoint.`
- what remains for Stage 5.11 consolidation? `Consolidate the now-complete Stage 5 training + regular eval + blind eval evidence into the final summary layer, and optionally note the small eval-config snapshot consistency fix landed after this audit.`
