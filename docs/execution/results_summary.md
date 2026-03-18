# Results Summary

汇总当前 Stage 3 已经获得的事实性结果，并明确哪些结论已经可以说、哪些还不能说。

## 当前阶段

- Stage 4: Minimal Closed-Loop Learnability (**PASS**)
- 当前定位：
  - 已完成 Stage 4 单样本 overfit 验收（PASS）
  - 已完成 Stage 4 小子集验收（PASS，带保留项）
  - 下一阶段为 Stage 5 paper-aligned setting alignment（GO）
  - 不代表论文指标已达成

## 当前已验证的事实

### 1. Stage 3 optical contract 已落地

- 当前主链路已经稳定到：
  - `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- full-grid forward 与 ROI supervision 已显式分离
- `U_out_full`、`I_out_full`、`I_out_roi` 都由 optical module 直接返回

### 2. Readout / Crop 最小验证通过

来源：
- `scripts/check_optical_readout.py`

当前重跑结果：
- `U_out_full` 为 complex tensor
- `I_out_full` 和 `I_out_roi` 为 real 且 nonnegative
- `output_crop_hw=(20, 20)` 时 ROI shape 为 `(1, 1, 20, 20)`
- `output_crop_hw=(12, 12)` 时 ROI shape 为 `(1, 1, 12, 12)`
- oversized crop 会报显式错误
- 改变输入 phase 会改变输出：
  - `full_delta = 1.748107e-03`
  - `roi_delta = 2.850121e-03`

### 3. Forward sanity 已覆盖 L=1/3/5

来源：
- `scripts/check_optical_forward_depths.py`

当前重跑结果：
- `L=1/3/5` 都能 forward 成功
- 三个 depth 的输出接口一致：
  - `U0`
  - `U_out_full`
  - `I_out_full`
  - `I_out_roi`
- `U_out_full` 持续保持 complex
- `I_out_full` 与 `I_out_roi` 持续保持 real 且 nonnegative
- 错误的 `inter_layer` 长度会被显式拒绝

### 4. Decoder-only single-sample fitting 已显示明确下降

来源：
- `outputs/optics/decoder_only_single_sample_smoke/summary.json`

当前 `L=3` smoke 结果：

| depth | steps | initial_loss | best_loss | final_loss | loss_decrease |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 120 | 0.1463065892457962 | 0.03812457248568535 | 0.03893868252635002 | 0.10818201676011086 |

补充观察：
- `roi_change_mean_abs = 0.4759185314178467`
- `best_roi_std = 0.34699586033821106`

这说明在不引入 encoder 的条件下，当前 optical decoder stack 至少能对一个目标做出可观下降。

### 5. Fixed-protocol small-subset depth sweep 已覆盖 L=1/3/5

来源：
- `outputs/optics/decoder_only_small_subset_sweep_run1/sweep_summary.json`
- `outputs/optics/decoder_only_small_subset_sweep_run1/depth_comparison.csv`

shared protocol：
- `subset_size = 4`
- `steps = 80`
- `seed = 42`
- `device = cpu`
- `lr_phase = 0.15`
- `lr_decoder = 0.03`
- `freeze_decoder = false`
- `loss = normalized_mae_roi`

结果摘要：

| depth | success_count | total_count | success_rate | mean_initial_loss | mean_best_loss | mean_loss_decrease |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 4 | 1.0 | 0.12581018544733524 | 0.02864284673705697 | 0.09716733871027827 |
| 3 | 4 | 4 | 1.0 | 0.13204407505691051 | 0.03225723281502724 | 0.09978684224188328 |
| 5 | 4 | 4 | 1.0 | 0.1348664928227663 | 0.03135538613423705 | 0.10351110668852925 |

最小事实结论：
- 三种 depth 在同一 tiny protocol 下都能在多个样本上下降
- Issue 6 需要的 fixed-protocol `L=1/3/5` 容量 sweep 已补齐

## 当前可以说的结论

- 当前 Stage 3 optical decoder skeleton 是可运行的，且 optical/readout/crop contract 已稳定。
- 在当前 tiny synthetic protocol 下，`L=1/3/5` 三种 depth 都表现出可优化的 decoder-only capacity。
- 当前结果已经足以支撑：
  - 进入未来 Stage 4 时不必重新设计 optical core
  - future encoder 只需要对接 phase-provider / `forward_from_phase(...)` contract

## 当前不能说的结论

- 不能说论文性能已经复现。
- 不能说 `L=5` 或任何更深 depth 已经被严格证明更优。
- 不能说当前 tiny synthetic subset 结果可直接外推到真实数据集或自然图像。
- 不能说当前 optical module 已经完成 final paper-setting alignment。
- 不能说 Stage 4 end-to-end training 已经就绪到可以跳过进一步调试。

## 已知限制与风险

- 当前验证协议使用的是 Stage 3 工程化 tiny setting，不是 final paper full-setting。
- 当前子集目标是 deterministic synthetic ROI targets，主要用于 capacity sanity，不是正式 benchmark。
- 当前 sweep 在 CPU 上完成，默认使用 float32 计算与较小 grid。
- propagation primitive 来自 upstream optics 思路的重写版本，当前已做 forward / fitting sanity，但尚未扩展到硬件鲁棒性或更严格物理对照。
- current depth comparison 只说明“在相同预算下都能下降”，不说明更深层必然更好。

## 后续阶段边界

- Stage 4 才进入 encoder 接入与最小闭环训练。
- Stage 5/6 才进入 paper-final setting alignment、更多数据协议和系统化实验。

---

## Stage 4 Closed-Loop Learnability Summary

### �ѽ�������ʵ

- �������ջ� loss ��ȷ�½���encoder �� optics �ݶȿɹ۲⣬�� NaN/Inf
- С�Ӽ��ջ� loss �����½����������������Ӧ�������� collapse
- Stage 4 learnability gate ������PASS��

### ��δ��������ʵ

- ���ļ�����������ָ�꣨PSNR/SSIM ��ƫ�ͣ�
- paper-final setting alignment
- ���ģѵ����ϵͳ������

### ��Ҫ������

- С�Ӽ�������ƫģ����blob-like
- normalized MAE �� loss ƽֵ̨�ϸ�
- ����ԭ�������С encoder ������toy optics grid��Ŀ���һ����ʽ�ȣ��� Stage 5 ��һ����֤��

### Stage 5 ����

- ���ۣ�**GO**
- ���ɣ�Stage 4 learnability gate ��ͨ��������������δ�������趨��Stage 5 �������Ǳ�Ҫ����
