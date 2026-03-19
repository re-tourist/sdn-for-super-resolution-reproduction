执行了一个真实的 Stage 5 paper-aligned smoke run，并把 smoke-only 预算、训练稳定性、artifact 路径、regular eval 跟进和 blind eval 跟进都收口成了单独报告，没有把这次短跑伪装成 main run，也没有重开 trainer / eval 协议。

精确变更文件：
- [configs/stage5/stage5_trainer_smoke.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_trainer_smoke.yaml)
- [docs/execution/stage5_smoke_report.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/docs/execution/stage5_smoke_report.md)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)
- [docs/ai/CodexFeedback/stage5/feedback_stage5-9.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/docs/ai/CodexFeedback/stage5/feedback_stage5-9.md)

smoke config 与命令：
- smoke config: [configs/stage5/stage5_trainer_smoke.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_trainer_smoke.yaml)
- smoke-only budget:
  - `depth = 5`
  - `device = cpu`
  - `steps = 4`
  - `batch_size = 2`
  - `train_subset_size = 12`
  - `val_subset_size = 4`
  - `validate_every = 1`
  - `preview_limit = 2`
  - `resume = no`
- training command:
  - `python scripts/train_stage5_paper.py --config configs/stage5/stage5_trainer_smoke.yaml --run-name issue5_9_l5_smoke`

实际产出的 smoke artifacts：
- smoke run root:
  - [outputs/stage5/smoke/issue5_9_l5_smoke](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke)
- key files:
  - [config_snapshot.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/config_snapshot.json)
  - [history.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/history.json)
  - [grad_stats.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/grad_stats.json)
  - [run_summary.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/run_summary.json)
  - [loss_curve.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/loss_curve.png)
  - [preview_step0.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/preview_step0.png)
  - [preview_best.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/preview_best.png)
  - [preview_final.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/preview_final.png)
  - [checkpoint_best.pt](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt)
  - [checkpoint_latest.pt](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_latest.pt)

关键稳定性结论：
- 训练在 `L=5`、`4` 步、CPU smoke 预算下顺利完成，无 NaN / Inf
- encoder 和 decoder 在 `1~4` 步都存在非零梯度
- `val_loss` 从 `0.0702461712` 下降到 `0.0702020377`
- `train_loss` 从 `0.1281592697` 下降到 `0.0876854211`，中间有小幅波动，但始终有限且可解释
- `best_step = 4`，`best_val_loss = 0.0702020377`
- 因为冻结的 Stage 5 `gamma_by_depth[5] = 0.0`，本次 smoke 的 efficiency term 记录为 `0.0`，这是沿用 `5.5`，不是本 issue 新决定

smoke 之后是否执行 eval：
- regular eval：已执行
  - command:
    - `python scripts/eval_stage5_paper.py --config configs/stage5/stage5_eval.yaml --checkpoint outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt --split val --subset-size 4 --run-name issue5_9_smoke_val_eval`
  - outputs:
    - [summary.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/issue5_9_smoke_val_eval/summary.json)
    - [preview_comparison.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/issue5_9_smoke_val_eval/preview_comparison.png)
  - metrics:
    - model `PSNR = 13.0716336177`
    - model `SSIM = 0.2514762123`
    - bicubic `PSNR = 28.4709274795`
    - bicubic `SSIM = 0.9607577262`
- blind eval：已执行
  - command:
    - `python scripts/eval_stage5_blind_linepair.py --config configs/stage5/stage5_blind_eval.yaml --checkpoint outputs/stage5/smoke/issue5_9_l5_smoke/checkpoints/checkpoint_best.pt --run-name issue5_9_smoke_blind_eval`
  - outputs:
    - [summary.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/summary.json)
    - [blind_targets_grid.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/blind_targets_grid.png)
    - [blind_outputs_grid.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/blind_outputs_grid.png)
    - [blind_comparison_grid.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_9_smoke_blind_eval/blind_comparison_grid.png)
  - blind target bank size: `24`

留给 Issue 5.10 的内容：
- 单独定义 main-run budget，不能复用这次 smoke config
- 更长训练和正式 val/test 记录
- main-run 层面的结果报告与复盘
- 继续消费现有 trainer / regular eval / blind eval 路径，而不是改写它们

推荐的 commit 三段式信息：

```text
feat(train | stage5-9): run paper-aligned smoke training

why:
validate that the frozen stage5 paper-path can train end to end under a
small smoke-only budget before main-run work, and record stability plus
artifact completeness without confusing smoke with final results

what:
add a dedicated stage5 smoke config, execute an L=5 smoke training run,
record checkpoints, history, previews, and a smoke report, and attach
post-run regular eval and blind eval evidence using the existing stage5
paths
```
