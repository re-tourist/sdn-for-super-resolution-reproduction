实现了一个独立于 regular eval 的 Stage 5 blind line-pair 路径，职责只覆盖 deterministic blind target 生成、复用现有 Stage 5 checkpoint/model forward，以及保存可审计的 blind artifact；没有把 5.7 的 PSNR/SSIM + bicubic baseline 逻辑和 5.8 混成一个大脚本，也没有碰 trainer、crop 语义或 Stage 6 研究项。

精确变更文件：
- [src/eval/stage5_blind_targets.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/eval/stage5_blind_targets.py)
- [scripts/eval_stage5_blind_linepair.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/eval_stage5_blind_linepair.py)
- [configs/stage5/stage5_blind_eval.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_blind_eval.yaml)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)

blind targets 如何生成：
- 默认 target family 为 `line_pair`
- 画布固定为 `96x96`
- 生成方式是 centered binary line-pair，不依赖 EMNIST dataset path
- 默认参数集：
  - `orientations = [horizontal, vertical]`
  - `line_widths_px = [1, 2]`
  - `gap_px = [1, 2, 3, 4, 6, 8]`
  - `margin_px = 12`
  - `line_length_px = 72`
  - `render_mode = binary`
  - `blur_sigma = 0.0`
- 因此默认会稳定生成 `2 x 2 x 6 = 24` 个 blind line-pair targets
- 这些值在 [stage5_blind_eval.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_blind_eval.yaml) 里可配置，并明确标注为 Stage 5 engineering default，而不是 paper-unique truth

blind-test hook 如何复用 Stage 5 model/eval 路径：
- 从 `5.6` 的 checkpoint 直接加载 `encoder_state_dict`、`decoder_state_dict` 和 `config_snapshot`
- 用 checkpoint 内嵌的 `encoder_config / optics_config / depth` 还原 `PaperPhaseEncoder + DiffractiveDecoder`
- 生成的 blind target 直接作为 `96x96` 单通道 HR 输入喂给 encoder
- 继续复用冻结的 `encoder -> decoder -> I_out_roi` 前向链
- 继续复用 `Stage4RoiTargetAdapter` 生成 `target_roi`，保证 crop/FOV 语义和 regular eval 一致
- 本 issue 不引入新的 paper-final scalar blind metric，只保存 artifacts 与 per-target metadata 供后续复盘

本次执行的 sanity 命令：
- `python -m py_compile src/eval/stage5_blind_targets.py scripts/eval_stage5_blind_linepair.py`
- `python scripts/eval_stage5_blind_linepair.py --config configs/stage5/stage5_blind_eval.yaml --checkpoint outputs/stage5/paper_trainer/issue5_6_resume_demo/checkpoints/checkpoint_best.pt --run-name issue5_8_blind_sanity`

已生成的 artifacts：
- [config snapshot](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_8_blind_sanity/config_snapshot.json)
- [summary](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_8_blind_sanity/summary.json)
- [blind target grid](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_8_blind_sanity/blind_targets_grid.png)
- [blind output grid](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_8_blind_sanity/blind_outputs_grid.png)
- [blind comparison grid](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/blind_eval/issue5_8_blind_sanity/blind_comparison_grid.png)

这次 sanity 记录到的关键事实：
- blind target bank 大小为 `24`
- 输入 shape 为 `(24, 1, 96, 96)`
- encoder 输出 `phi_lr` shape 为 `(24, 1, 32, 32)`
- sample forward shape 记录为：
  - `U0 = (4, 1, 400, 400)`
  - `U_out_full = (4, 1, 400, 400)`
  - `I_out_full = (4, 1, 400, 400)`
  - `I_out_roi = (4, 1, 96, 96)`
- crop 假设继续沿用 `5.3`：`output_crop_hw = [96, 96]`、`crop_policy = center_crop`、`center_offset_hw = [152, 152]`

有意留给后续 issue 的内容：
- regular eval 的 PSNR / SSIM / bicubic baseline 仍留在 `5.7`，没有在这里重写
- blind line-pair 之外更宽的 blind resolution-target 族扩展没有在这里做成“大而全”框架
- 任何量化、错位、鲁棒性、ablation 都仍归 Stage 6
- smoke/main run 编排仍归 `5.9` / `5.10`

推荐的 commit 三段式信息：

```text
feat(eval | stage5-8): add blind line-pair eval hook

why:
land a separate stage5 blind-test path after regular eval so line-pair
artifacts can be generated and inspected reproducibly without rewriting
PSNR/SSIM evaluation or pulling in stage6 study scope

what:
add a deterministic blind line-pair target generator, a stage5 blind
checkpoint-eval script that reuses the existing encoder-decoder forward
path, a configurable blind-eval config, and saved blind target/output
artifacts plus metadata under outputs/stage5/blind_eval
```
