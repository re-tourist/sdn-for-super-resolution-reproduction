# Stage 5 Smoke Report

## 1. Objective And Scope Boundary

This report records one real Stage 5 paper-aligned smoke run.

The purpose of this issue is narrow:
- validate that the frozen Stage 5 pipeline can train end to end
- confirm that artifacts and post-run evaluation hooks are usable
- keep smoke clearly separate from the later Stage 5.10 main run

This report does **not** claim a final paper result, a tuned budget, or a main-run-quality metric.

---

## 2. Smoke Config And Budget

Smoke config:
- `configs/stage5/stage5_trainer_smoke.yaml`

Smoke-only budget choices:
- depth: `L=5`
- device: `cpu`
- total steps: `4`
- batch size: `2`
- validate every: `1`
- train subset size: `12`
- val subset size: `4`
- preview limit: `2`
- resume used: `no`

Important boundary note:
- this reduced budget is a **smoke-only validation choice**
- it is not the Stage 5 main-run budget and does not replace the frozen paper-aligned main target of longer training

---

## 3. Commands Executed

Smoke training:

```bash
python scripts/train_stage5_paper.py --config configs/stage5/stage5_trainer_smoke.yaml --run-name issue5_9_l5_smoke
```

Post-run regular eval:

```bash
python scripts/eval_stage5_paper.py --config configs/stage5/stage5_eval.yaml --checkpoint outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt --split val --subset-size 4 --run-name issue5_9_smoke_val_eval
```

Post-run blind eval:

```bash
python scripts/eval_stage5_blind_linepair.py --config configs/stage5/stage5_blind_eval.yaml --checkpoint outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt --run-name issue5_9_smoke_blind_eval
```

---

## 4. Smoke Run Outputs

Smoke run directory:
- `outputs/stage5/smoke/issue5_9_l5_smoke`

Key artifacts:
- `outputs/stage5/smoke/issue5_9_l5_smoke/config_snapshot.json`
- `outputs/stage5/smoke/issue5_9_l5_smoke/history.json`
- `outputs/stage5/smoke/issue5_9_l5_smoke/grad_stats.json`
- `outputs/stage5/smoke/issue5_9_l5_smoke/loss_curve.png`
- `outputs/stage5/smoke/issue5_9_l5_smoke/run_summary.json`
- `outputs/stage5/smoke/issue5_9_l5_smoke/preview_step0.png`
- `outputs/stage5/smoke/issue5_9_l5_smoke/preview_best.png`
- `outputs/stage5/smoke/issue5_9_l5_smoke/preview_final.png`
- `outputs/stage5/smoke/issue5_9_l5_smoke/phi_preview_step0.png`
- `outputs/stage5/smoke/issue5_9_l5_smoke/phi_preview_best.png`
- `outputs/stage5/smoke/issue5_9_l5_smoke/phi_preview_final.png`
- `outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_latest.pt`
- `outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt`

Regular-eval follow-up artifacts:
- `outputs/stage5/eval/issue5_9_smoke_val_eval/config_snapshot.json`
- `outputs/stage5/eval/issue5_9_smoke_val_eval/summary.json`
- `outputs/stage5/eval/issue5_9_smoke_val_eval/preview_comparison.png`

Blind-eval follow-up artifacts:
- `outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/config_snapshot.json`
- `outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/summary.json`
- `outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/blind_targets_grid.png`
- `outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/blind_outputs_grid.png`
- `outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/blind_comparison_grid.png`

---

## 5. Stability And Loss Trend

### 5.1 Training Stability

Observed outcome:
- training completed without NaN / Inf
- checkpoints, previews, history, and summaries were all written successfully
- encoder and decoder gradients were present and non-zero at every recorded step

Gradient evidence from `grad_stats.json`:
- encoder: `grad_tensor_count = 14 / nonzero_grad_tensor_count = 14` for steps `1~4`
- decoder: `grad_tensor_count = 1 / nonzero_grad_tensor_count = 1` for steps `1~4`

### 5.2 Loss Trend

Recorded losses from `history.json`:
- step 1: `train_loss = 0.1281592697`, `val_loss = 0.0702461712`
- step 2: `train_loss = 0.0773873404`, `val_loss = 0.0702302046`
- step 3: `train_loss = 0.0935746357`, `val_loss = 0.0702152960`
- step 4: `train_loss = 0.0876854211`, `val_loss = 0.0702020377`

Interpretation:
- train loss is noisy at this tiny smoke budget, but remains finite and within a narrow, interpretable range
- val loss improves monotonically across the 4 recorded validation points
- best checkpoint is the final checkpoint at step `4`

### 5.3 Efficiency-Term Observation

Current smoke config consumes the frozen Stage 5 loss path with `efficiency_term.enabled = true`, but under the frozen Stage 5 default for `L=5`:
- `gamma_by_depth[5] = 0.0`

So in this smoke run:
- `train_efficiency_term = 0.0`
- `val_efficiency_term = 0.0`

This is expected Stage 5 behavior, not a smoke-specific redefinition.

---

## 6. Post-Run Eval Follow-Up

### 6.1 Regular Eval

Runner:
- `scripts/eval_stage5_paper.py`

Checkpoint:
- `outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt`

Evaluated split:
- `val`

Evaluated subset size:
- `4`

Recorded metrics:
- model PSNR: `13.0716336177`
- model SSIM: `0.2514762123`
- bicubic PSNR: `28.4709274795`
- bicubic SSIM: `0.9607577262`

Interpretation:
- this smoke checkpoint is far from a competitive result, which is expected after only 4 smoke steps
- the value of this follow-up is protocol validation: the checkpoint loads, the frozen ROI-aligned target path scores correctly, and the bicubic baseline comparison is reproducible

### 6.2 Blind Eval

Runner:
- `scripts/eval_stage5_blind_linepair.py`

Checkpoint:
- `outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt`

Recorded facts:
- blind target family: `line_pair`
- blind target count: `24`
- blind input shape: `(24, 1, 96, 96)`
- blind `phi_lr` shape: `(24, 1, 32, 32)`
- sample forward shape: `U0 / U_out_full / I_out_full / I_out_roi = (4,1,400,400) / (4,1,400,400) / (4,1,400,400) / (4,1,96,96)`

Interpretation:
- blind-eval hook works on the smoke checkpoint without changing crop or model semantics
- Issue 5.8's intended artifact path is usable for later reporting

---

## 7. Conclusion For Issue 5.9

Conclusion:
- this smoke run is sufficient to show that the Stage 5 paper-aligned pipeline is launchable, stable, and inspectable under a modest smoke-only budget
- it unblocks Issue 5.10 from an engineering-readiness perspective

What remains for Issue 5.10:
- launch a clearly separate main-run budget
- keep main-run config / command / report distinct from this smoke config
- run longer training and record main-run val/test results
- compare main-run outputs against the existing regular-eval and blind-eval paths, without redefining them

In short:
- **5.9 validates pipeline stability and artifact completeness**
- **5.10 is still required for a real paper-aligned main run**
