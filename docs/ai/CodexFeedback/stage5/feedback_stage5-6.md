实现了一个 Stage 5 专用的 paper-aligned trainer 路径，没有把它扩成通用训练框架。核心落地是新增 `scripts/train_stage5_paper.py`，并配套 `configs/stage5/stage5_trainer_short.yaml`，让当前仓库能把 `Stage 5 dataset + encoder + diffractive decoder + Stage5SuperResolutionLoss` 串起来跑一个可审计的短程训练与断点续跑。

精确变更文件：
- [scripts/train_stage5_paper.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/train_stage5_paper.py)
- [configs/stage5/stage5_trainer_short.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_trainer_short.yaml)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)

trainer 组装方式：
- dataset：复用 `build_stage5_emnist_display_dataset(...)`，显式消费 Stage 5 已冻结的 EMNIST `96x96` display protocol
- target 适配：复用 `Stage4RoiTargetAdapter`，按 decoder 的 `readout_config.output_crop_hw` 生成 `target_roi`
- model：直接组装 `PaperPhaseEncoder + DiffractiveDecoder`
- loss：直接调用 `Stage5SuperResolutionLoss`
- optimizer：显式分成两组 parameter groups
  - encoder LR = `0.0005`
  - decoder LR = `0.001`

本次拍板的 trainer-side `input_power` 来源：
- 选择 `U0_full_grid_intensity_sum`
- 具体实现为：对 decoder `forward_from_phase(...)` 返回的 `U0` 计算 `|U0|^2`，再在所有非 batch 维上求和
- 也就是：
  - `input_power = sum(|U0|^2)` per sample
- 这个选择已经同时写进：
  - [stage5_trainer_short.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_trainer_short.yaml)
  - [config_snapshot.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/config_snapshot.json)
  - [run_summary.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/run_summary.json)

artifact 产物：
- [config_snapshot.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/config_snapshot.json)
- [history.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/history.json)
- [grad_stats.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/grad_stats.json)
- [loss_curve.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/loss_curve.png)
- [preview_step0.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/preview_step0.png)
- [preview_best.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/preview_best.png)
- [preview_final.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/preview_final.png)
- [preview_resume_start.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/preview_resume_start.png)
- [phi_preview_step0.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/phi_preview_step0.png)
- [phi_preview_best.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/phi_preview_best.png)
- [phi_preview_final.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/phi_preview_final.png)
- [phi_preview_resume_start.png](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/phi_preview_resume_start.png)
- [checkpoint_latest.pt](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/checkpoints/checkpoint_latest.pt)
- [checkpoint_best.pt](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/paper_trainer/issue5_6_resume_demo/checkpoints/checkpoint_best.pt)

本次执行的 short-run sanity 命令：
- `python -m py_compile scripts/train_stage5_paper.py`
- `python scripts/train_stage5_paper.py --config configs/stage5/stage5_trainer_short.yaml --run-name issue5_6_resume_demo --steps 1`
- `python scripts/train_stage5_paper.py --config configs/stage5/stage5_trainer_short.yaml --run-name issue5_6_resume_demo --steps 2 --resume outputs/stage5/paper_trainer/issue5_6_resume_demo/checkpoints/checkpoint_latest.pt`

sanity evidence：
- short run 完成到 step 2，且 resume 成功从 step 1 续跑到 step 2
- `history.json` 里保留了 step 1 和 step 2 两条记录
- `run_summary.json` 记录了：
  - `encoder_input = [2, 1, 96, 96]`
  - `phi_lr = [2, 1, 32, 32]`
  - `U0 = [2, 1, 400, 400]`
  - `I_out_full = [2, 1, 400, 400]`
  - `I_out_roi = [2, 1, 96, 96]`
- `best_val_loss = 0.06892300397157669`
- `final_val_loss = 0.06892351806163788`

有意留给后续 issue 的未决项：
- Stage 5 smoke run / main run 的正式训练步数、运行预算与实验排程，仍留给后续 smoke/main run issue
- eval runner、PSNR/SSIM、bicubic baseline、blind line-pair protocol 仍不在本 issue 内实现
- dataset tiling、optics distance mapping、phase range、gamma policy、sigma policy 都继续沿用 5.2 到 5.5 的冻结结果，这个 trainer 不重开它们

推荐的 commit 三段式信息：

```text
feat(train | stage5-6): add stage5 paper trainer

why:
land a stage5-specific end-to-end trainer before smoke and main runs so
the frozen dataset, optics, encoder, and loss configs can be consumed by
a reproducible path with explicit resume and artifact behavior

what:
add a stage5 paper trainer script and short-run config, wire dataset,
encoder, decoder, target adapter, and stage5 sr loss together, use
U0_full_grid_intensity_sum as the explicit input_power source, and save
checkpoints, history, summaries, and previews for sanity and resume
validation
```
