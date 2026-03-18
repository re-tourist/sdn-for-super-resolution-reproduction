# Results Summary

本文档汇总当前仓库已经获得的事实性结果，用于回答两个问题：

1. 现在已经被验证的工程结论是什么。
2. 现在还不能声称已经完成了什么。

本文档是结果摘要，不是实现说明书，也不是后续阶段计划文档。

---

## 当前阶段

- 当前可信阶段结论：`Stage 4 minimal closed-loop learnability = PASS`
- 当前工程状态：
  - Stage 3 optical module verification 已完成
  - Stage 4 单样本 closed-loop 验收已完成
  - Stage 4 小子集 closed-loop 验收已完成（带保留项）
  - Stage 5 paper-aligned planning 已起草，但 paper-aligned implementation 尚未开始

---

## 当前已验证的事实

### 1. Stage 3 optical contract 已冻结并落地

- 当前主链路稳定为：
  - `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- full-grid forward 与 ROI supervision 已显式分离
- `U_out_full`、`I_out_full`、`I_out_roi` 由 optical module 直接返回

### 2. Readout / crop / forward sanity 已通过

来源：

- `scripts/check_optical_readout.py`
- `scripts/check_optical_forward_depths.py`

最小事实：

- `I_out_full` / `I_out_roi` 保持 real 且 nonnegative
- `U_out_full` 保持 complex
- `output_crop_hw` 越界会显式报错
- `L=1/3/5` 在当前 Stage 3 tiny setting 下都可 forward

### 3. Decoder-only optical capacity 已通过最小验证

来源：

- `outputs/optics/decoder_only_single_sample_smoke/summary.json`
- `outputs/optics/decoder_only_small_subset_sweep_run1/sweep_summary.json`

最小事实：

- decoder-only 单样本拟合中，loss 可明确下降
- decoder-only 小子集 fixed-protocol sweep 中，`L=1/3/5` 都能在多个样本上下降
- 这些结果说明 optical decoder stack 具备可优化容量
- 这些结果不等价于 paper-aligned end-to-end 性能已经成立

### 4. Stage 4 单样本 closed-loop 验收已通过

来源：

- `docs/execution/stage4_single_sample_report.md`
- `outputs/stage4/minimal_trainer/issue4_6_single_sample_100/run_summary.json`
- `outputs/stage4/minimal_trainer/issue4_6_single_sample_300/run_summary.json`

结果摘要：

| run | steps | initial_loss | best_loss | final_loss | encoder grad | optics grad | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `issue4_6_single_sample_100` | 100 | 0.213564 | 0.164205 | 0.164152 | observed | observed | none |
| `issue4_6_single_sample_300` | 300 | 0.213564 | 0.117267 | 0.116939 | observed | observed | none |

最小事实：

- loss 明确下降
- encoder / optics 梯度全程可观测
- 无 NaN / Inf
- 预测 ROI 从初始扩散亮斑演化到与 target 更接近的结构

### 5. Stage 4 小子集 closed-loop 验收已通过

来源：

- `docs/execution/stage4_small_subset_report.md`
- `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s200/run_summary.json`
- `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s400/run_summary.json`

结果摘要：

| run | steps | initial_loss | best_loss | final_loss | val_loss | encoder grad | optics grad | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `issue4_7_small_subset_16_s200` | 200 | 0.217708 | 0.149736 | 0.180445 | 0.192186 | observed | observed | PASS |
| `issue4_7_small_subset_16_s400` | 400 | 0.217708 | 0.144116 | 0.171176 | 0.190612 | observed | observed | PASS |

最小事实：

- 小子集上 train loss 明确下降
- val loss 从初始状态下降后进入平台
- 输出对输入仍有响应，没有明显 collapse 到单一模式
- 无 NaN / Inf

---

## 当前可以说的结论

- 当前仓库已经通过 Stage 4 learnability gate。
- 现有 optical core 不需要在 Stage 5 之前重写。
- Stage 5 应把重点放在 paper-aligned dataset / optics config / encoder / loss / eval，而不是重新证明“系统能不能学”。
- Stage 4 的单样本和小子集工件应被保留为后续 Stage 5 回归基线。

---

## 当前不能说的结论

- 不能说论文指标已经复现。
- 不能说 paper-aligned pipeline 已经落地。
- 不能说 `L=5` 或任何更深 depth 已被严格证明更优。
- 不能说当前 Stage 4 结果可直接外推为 paper-final 质量。
- 不能说 blind line-pair、量化、鲁棒性或系统化消融已经完成。

---

## 已知限制与风险

- 当前 Stage 4 使用的是最小闭环协议，不是 paper-final setting。
- 当前 raw 读出质量仍偏模糊，PSNR / SSIM 偏低。
- Stage 4 的主损失是 normalized MAE；其优化目标与 raw 强度可视化 / PSNR / SSIM 不完全同向。
- 400x400 propagation grid、paper dataset protocol、efficiency penalty 和 line-pair eval 尚未进入正式实现。

---

## 对 Stage 5 的含义

- 结论：`GO`
- 解释：
  - Stage 4 已经回答“encoder + optics 是否可学”
  - Stage 5 的任务不再是 learnability gate，而是把系统对齐到论文设定
  - Stage 5 仍需对未决项保持诚实记录，不能把假设伪装成论文已明确

---

## 相关文档

- `docs/execution/stage4_single_sample_report.md`
- `docs/execution/stage4_small_subset_report.md`
- `docs/plan/stage_plan/stage5/stage5_plan.md`
- `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
