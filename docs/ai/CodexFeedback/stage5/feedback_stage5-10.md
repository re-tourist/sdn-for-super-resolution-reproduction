基于你从 Linux 服务器回传的完整 Stage 5 主训练与评估产物，Issue 5.10 现在已经完成到“主训练 + regular eval + blind eval 都有真实 artifact 和正式报告”的状态。本次工作不仅补全了主报告，也顺手修正了 eval / blind eval 的一个小审计一致性问题：未来生成的 `config_snapshot.json` 会记录实际生效的 runtime 参数，而不再出现顶部 `device = cuda` 但 `runtime.device = cpu` 这种由 config 默认值残留导致的混淆。

精确变更文件：
- [stage5_main_run_report.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/docs/execution/stage5_main_run_report.md)
- [feedback_stage5-10.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/docs/ai/CodexFeedback/stage5/feedback_stage5-10.md)
- [eval_stage5_paper.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/eval_stage5_paper.py)
- [eval_stage5_blind_linepair.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/eval_stage5_blind_linepair.py)

本次确认并写入正式报告的主训练事实：
- run name：`stage5_l5_phase_main`
- main config：`configs/stage5/stage5_trainer_main.yaml`
- 训练命令：
  - `python scripts/train_stage5_paper.py --config configs/stage5/stage5_trainer_main.yaml --run-name stage5_l5_phase_main --device cuda`
- 实际完成：
  - `completed_steps = 750000`
  - `steps_per_epoch = 1500`
  - 对应 `500` epochs
- 核心数值：
  - `best_step = 745500`
  - `best_val_loss = 0.0188692901`
  - `final_train_loss = 0.0201732814`
  - `final_val_loss = 0.0188748985`
- 运行状态：
  - `resume_from = null`
  - `phase = completed`
  - 未看到 NaN / Inf 终止痕迹
- 时间范围：
  - start: `2026-03-19T07:00:40.685304+00:00`
  - end: `2026-03-19T13:19:59.703498+00:00`
  - wall-clock: `6h 19m 19s`

本次确认到的 regular eval 结果：
- val eval root：
  - [stage5_l5_phase_main_val_eval](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/stage5_l5_phase_main_val_eval)
- val metrics：
  - model `PSNR = 14.9180516516`
  - model `SSIM = 0.8126175770`
  - bicubic `PSNR = 25.6875014105`
  - bicubic `SSIM = 0.9447924524`
- test eval root：
  - [stage5_l5_phase_main_test_eval](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/stage5_l5_phase_main_test_eval)
- test metrics：
  - model `PSNR = 9.5773711626`
  - model `SSIM = 0.4385546151`
  - bicubic `PSNR = 20.2487178734`
  - bicubic `SSIM = 0.8392536711`
- 这两次 eval 都对齐到同一个主训练 best checkpoint：
  - `outputs/stage5/main_run/stage5_l5_phase_main/checkpoints/checkpoint_best.pt`

本次确认到的 blind eval 结果：
- blind eval root：
  - [stage5_l5_phase_main_blind_eval](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval)
- blind target family：`line_pair`
- blind target count：`24`
- 关键 blind artifact：
  - [blind_targets_grid.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_targets_grid.png)
  - [blind_outputs_grid.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_outputs_grid.png)
  - [blind_comparison_grid.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/blind_comparison_grid.png)
- 审计结论：
  - blind hook 已对同一个 best checkpoint 跑通
  - `24` 个 deterministic line-pair targets 都有输出
  - 但 `per_target_records` 里模型输出能量整体远低于二值目标能量，因此不应把这次 blind eval 解释成“已经证明了强 blind-resolution 表现”；仍应以保存下来的 grids 做人工检查与 Stage 5 总结

这次顺手修掉的审计一致性问题：
- 之前的 `eval_stage5_paper.py` 和 `eval_stage5_blind_linepair.py` 会在 `config_snapshot.json` 里保留 config 默认 runtime 值
- 所以会出现：
  - 顶层 `device = cuda`
  - 但 `runtime.device = cpu`
- 现在已经改成记录“实际生效的 runtime 参数”
- 这不会改变你这次已经上传回来的数值结果，但会避免后续再出现同类歧义

当前还剩什么工作：
- 从 `5.10` 自身来看，主训练和对应评估链条已经齐套，不再缺主干 artifact
- 留给 `5.11` 的主要是 consolidation：
  - 汇总 Stage 5 训练、regular eval、blind eval 的结论
  - 把工程事实和论文对齐结论收口成阶段总结

推荐的 commit 三段式信息：

```text
docs(execution | stage5-10): finalize main-run report and eval audit

why:
close out stage5 main-run reporting after the full linux-server
training, regular eval, and blind eval artifacts were uploaded, and fix
the remaining eval snapshot ambiguity so future audits record effective
runtime settings clearly

what:
fill the stage5 main run report with the uploaded L=5 training and eval
results, add the final stage5-10 feedback note, and update the eval and
blind-eval scripts so config snapshots store effective runtime values
instead of stale config defaults
```
