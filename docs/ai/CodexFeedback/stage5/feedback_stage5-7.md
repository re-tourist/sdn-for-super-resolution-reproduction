实现了一个 Stage 5 专用 regular eval 路径，职责只覆盖 `PSNR / SSIM + bicubic baseline`，没有把 blind line-pair、trainer refactor 或 Stage 6 对比实验掺进来。核心落地是新增 `scripts/eval_stage5_paper.py` 与 `configs/stage5/stage5_eval.yaml`，并让它能够直接加载 `5.6` 的 checkpoint、复用冻结的 ROI target 语义、输出可审计的 summary 与 preview artifact。

精确变更文件：
- [scripts/eval_stage5_paper.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/eval_stage5_paper.py)
- [configs/stage5/stage5_eval.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_eval.yaml)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)

Stage 5 regular eval 路径如何工作：
- 从 `--checkpoint` 加载 `5.6` 训练器保存的 `encoder_state_dict`、`decoder_state_dict` 和 `config_snapshot`
- 用 checkpoint 内嵌的 `encoder_config / optics_config / depth` 还原模型，不依赖外部 trainer refactor
- 从 [stage5_emnist_display.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_emnist_display.yaml) 构造 `val` 或 `test` 数据集
- 复用 `Stage4RoiTargetAdapter` 生成 `target_roi`
- 用当前模型路径计算 `prediction_roi = I_out_roi`
- 用现有 [evaluator.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/eval/evaluator.py) / [metrics.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/eval/metrics.py) 计算 PSNR / SSIM
- 同时生成 bicubic baseline 并对同一个 `target_roi` 记分

bicubic baseline 的实现口径：
- baseline source tensor：`target_roi`
- 默认 LR 尺寸：`32x32`
- 默认 scale factor：`3`
- 实现方式：`96x96 target_roi -> bicubic downsample -> 32x32 -> bicubic upsample -> 96x96`
- anti-aliasing：
  - downsample：`true`
  - upsample：`false`
- 插值模式：
  - downsample：`bicubic`
  - upsample：`bicubic`
- 这些假设已写入：
  - [stage5_eval.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_eval.yaml)
  - `outputs/stage5/eval/.../summary.json`

本次执行的 sanity 命令：
- `python -m py_compile scripts/eval_stage5_paper.py`
- `python scripts/eval_stage5_paper.py --config configs/stage5/stage5_eval.yaml --checkpoint outputs/stage5/paper_trainer/issue5_6_resume_demo/checkpoints/checkpoint_best.pt --split val --subset-size 4 --run-name issue5_7_val_sanity`
- `python scripts/eval_stage5_paper.py --config configs/stage5/stage5_eval.yaml --checkpoint outputs/stage5/paper_trainer/issue5_6_resume_demo/checkpoints/checkpoint_best.pt --split test --subset-size 4 --run-name issue5_7_test_sanity`

已产出的 eval artifacts：
- [val summary](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/issue5_7_val_sanity/summary.json)
- [val preview](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/issue5_7_val_sanity/preview_comparison.png)
- [test summary](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/issue5_7_test_sanity/summary.json)
- [test preview](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/eval/issue5_7_test_sanity/preview_comparison.png)

这次 sanity 的结果摘要：
- val 子集 4 样本：
  - model `PSNR = 16.2660`
  - model `SSIM = 0.8443`
  - bicubic `PSNR = 28.4709`
  - bicubic `SSIM = 0.9608`
- test 子集 4 样本：
  - model `PSNR = 9.6900`
  - model `SSIM = 0.4261`
  - bicubic `PSNR = 19.3473`
  - bicubic `SSIM = 0.8122`

有意留给 Issue 5.8 的内容：
- blind line-pair / resolution-target 生成
- blind test 的 artifact 结构与报告格式
- regular eval 之外的 blind-eval hook

推荐的 commit 三段式信息：

```text
feat(eval | stage5-7): add stage5 regular eval runner

why:
land a stage5-specific regular evaluation path before blind-test work so
paper-path checkpoints can be scored reproducibly on val/test with
explicit crop, normalization, and bicubic-baseline assumptions

what:
add a stage5 eval script and config, load stage5 checkpoints directly,
score I_out_roi against target_roi with PSNR/SSIM, generate a bicubic
downsample-then-upsample baseline with anti-aliasing, and save summaries
plus comparison previews under outputs/stage5/eval
```
