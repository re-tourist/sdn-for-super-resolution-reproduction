# Experiment Log

本文档按时间顺序记录当前仓库中已经实现并运行过的关键里程碑。

记录范围：

- Stage 3 optical module verification
- Stage 4 minimal closed-loop acceptance

不记录内容：

- 尚未执行的 Stage 5 paper-aligned implementation
- 计划文档中的未来任务
- 仅存在于讨论中、没有运行证据的结论

---

## S3-02 Optical Propagation Core

- 对应任务：Issue 2 `feat: extract paper-aligned propagation core from sdn_upstream`
- 做了什么：
  - 从 `external/sdn_upstream` 参考并重写 centered FFT / IFFT、Rayleigh-Sommerfeld transfer kernel、phase-mask modulation。
  - 在 `src/models/optics/propagation.py`、`src/models/optics/phase_utils.py`、`src/models/optics/diffractive_decoder.py` 中落地为当前 repo 的 clean optics core。
  - 去掉 classification head、electrical decoder 以及上游任务耦合逻辑。
- 最小结论：
  - optical core 已与 upstream 任务层解耦。
  - `L` 的语义固定为 trainable diffractive phase masks 数量。
  - distance schedule 已显式区分 `input_to_first / inter_layer / last_to_sensor`。

## S3-03 Readout / Crop Contract

- 对应任务：Issue 3 `feat: add output FOV crop and optical direct-readout contract`
- 做了什么：
  - 新增 `src/models/optics/readout.py`，把 `I_out_full = |U_out_full|^2` 和 deterministic center crop 从脚本中抽离。
  - decoder 稳定返回 `U_out_full`、`I_out_full`、`I_out_roi`。
  - `output_crop_hw` 成为显式配置，crop 越界会报错。
- 运行证据：
  - `scripts/check_optical_readout.py`
- 最小结论：
  - full-grid 与 ROI readout 的职责已稳定。
  - crop contract 已进入模块层，不再隐藏在脚本里。

## S3-04 Forward Sanity For L=1/3/5

- 对应任务：Issue 4 `feat: add Stage 3 forward sanity scripts for L=1/3/5`
- 做了什么：
  - 新增 `scripts/check_optical_forward_depths.py`
  - 在同一最小协议下检查 `L=1/3/5` 的 forward 行为
- 运行证据：
  - `scripts/check_optical_forward_depths.py`
- 最小结论：
  - `L=1/3/5` 都能成功 forward
  - 输出接口一致，invalid distance schedule 会显式失败

## S3-05 Decoder-Only Single-Sample Fitting

- 对应任务：Issue 5 `feat: add decoder-only single-sample fitting runner`
- 做了什么：
  - 新增 `scripts/train_decoder_only_single_sample.py`
  - 直接优化 learnable input phase，不引入 encoder
  - 在 ROI 上使用 normalized MAE 拟合单个 synthetic target
- 运行证据：
  - `outputs/optics/decoder_only_single_sample_smoke/summary.json`
- 最小结论：
  - decoder-only 单样本路径可学
  - 这只说明 optical decoder 有容量，不等价于 end-to-end 结果

## S3-06 Decoder-Only Small-Subset Depth Sweep

- 对应任务：Issue 6 `feat: run small-subset optical capacity experiment across L=1/3/5`
- 做了什么：
  - 新增 `scripts/train_decoder_only_small_subset.py`
  - 新增 `scripts/train_decoder_only_small_subset_sweep.py`
  - 对固定 deterministic subset 执行 `L=1/3/5` sweep
- 运行证据：
  - `outputs/optics/decoder_only_small_subset_sweep_run1/sweep_summary.json`
  - `outputs/optics/decoder_only_small_subset_sweep_run1/depth_comparison.csv`
- 最小结论：
  - 三种 depth 在同一 tiny protocol 下都能下降
  - 当前结果不构成 paper performance 排序结论

## S3-07 Phase-Provider Hook

- 对应任务：Task3-7 `feat: add clean phase-provider hook for future Stage 4 integration`
- 做了什么：
  - 新增 `src/models/optics/phase_provider.py`
  - 明确 `PhaseProvider` contract：`upstream_input -> phase tensor`
  - 在 decoder 中稳定 `forward_from_phase(...)`、`forward_from_field(...)`、`forward_from_phase_provider(...)`
- 最小结论：
  - future encoder 接入点已冻结
  - Stage 4 不需要重新设计 optical core 接口

---

## S4-01 Single-Sample Closed-Loop Acceptance

- 对应任务：Issue 4.6 `Run Stage 4 single-sample overfit sanity and write acceptance report`
- 做了什么：
  - 使用现有 `scripts/train_stage4_minimal.py` 执行正式单样本 overfit 验收
  - 未重写 optical core、dataset path、wrapper 或 trainer
- 运行命令：

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 100 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --download \
  --run-name issue4_6_single_sample_100
```

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 300 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_6_single_sample_300
```

- 工件目录：
  - `outputs/stage4/minimal_trainer/issue4_6_single_sample_100/`
  - `outputs/stage4/minimal_trainer/issue4_6_single_sample_300/`
- 结果摘要：

| run | steps | initial_loss | best_loss | final_loss | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `issue4_6_single_sample_100` | 100 | 0.213564 | 0.164205 | 0.164152 | PASS |
| `issue4_6_single_sample_300` | 300 | 0.213564 | 0.117267 | 0.116939 | PASS |

- 最小结论：
  - loss 明确下降
  - encoder / optics 梯度持续可观测
  - 无 NaN / Inf
  - 单样本 closed-loop learnability 成立
- 详细报告：
  - `docs/execution/stage4_single_sample_report.md`

## S4-02 Small-Subset Closed-Loop Acceptance

- 对应任务：Issue 4.7 `Run Stage 4 small-subset closed-loop training and summarize learnability`
- 做了什么：
  - 使用同一 Stage 4 最小训练栈运行真实小子集闭环训练
  - 保持 Stage 4 范围，不扩展到 Stage 5 paper settings
- 运行命令：

```bash
python scripts/train_stage4_minimal.py \
  --subset-size 16 \
  --batch-size 4 \
  --steps 200 \
  --preview-limit 4 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_7_small_subset_16_s200
```

```bash
python scripts/train_stage4_minimal.py \
  --subset-size 16 \
  --batch-size 4 \
  --steps 400 \
  --preview-limit 4 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_7_small_subset_16_s400
```

- 工件目录：
  - `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s200/`
  - `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s400/`
- 结果摘要：

| run | steps | initial_loss | best_loss | final_loss | val_loss | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `issue4_7_small_subset_16_s200` | 200 | 0.217708 | 0.149736 | 0.180445 | 0.192186 | PASS |
| `issue4_7_small_subset_16_s400` | 400 | 0.217708 | 0.144116 | 0.171176 | 0.190612 | PASS |

- 最小结论：
  - 小子集上 loss 下降
  - encoder / optics 梯度持续可观测
  - 输出随输入变化，没有明显 collapse
  - 输出质量仍偏模糊，PSNR / SSIM 偏低
- 详细报告：
  - `docs/execution/stage4_small_subset_report.md`

---

## 当前日志边界

- 本文档当前记录到 Stage 4 learnability acceptance 为止。
- Stage 5 目前只有 planning docs：
  - `docs/plan/stage_plan/stage5/stage5_plan.md`
  - `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
- 在尚未出现 paper-aligned 真实运行证据前，不应在本日志中提前写入 Stage 5 结果结论。
