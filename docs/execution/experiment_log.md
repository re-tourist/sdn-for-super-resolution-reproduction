# Experiment Log

记录 Stage 3 optical module verification 阶段的关键执行路径、最小验证结论和当前边界。

当前 Stage 3 冻结主链路：

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi -> loss/metrics`

本文档只记录当前仓库中已经实现并运行过的 Stage 3 里程碑，不把最小 smoke 或 tiny protocol 包装成最终论文结果。

## S3-02 Optical Propagation Core

- 对应任务：Issue 2 `feat: extract paper-aligned propagation core from sdn_upstream`
- 做了什么：
  - 从 `external/sdn_upstream` 参考并重写了 centered FFT / IFFT、Rayleigh-Sommerfeld transfer kernel、phase-mask modulation。
  - 在 `src/models/optics/propagation.py`、`src/models/optics/phase_utils.py`、`src/models/optics/diffractive_decoder.py` 内落地为当前 repo 的 clean optics core。
  - 去掉了 classification head、electrical decoder、Pet / OAM / LG / HG 任务耦合，以及 `layer == 2` 这类隐式 final propagation 逻辑。
- 为什么做：
  - Stage 3 需要先验证 optical decoder skeleton 本身可运行、可检查、可被后续 readout / crop / fitting 复用。
- 最小结论：
  - optical core 已独立于 upstream 任务层。
  - `L` 的语义已固定为 trainable diffractive phase masks 数量。
  - distance schedule 已改为显式 `input_to_first / inter_layer / last_to_sensor`。

## S3-03 Readout / Crop Contract

- 对应任务：Issue 3 `feat: add output FOV crop and optical direct-readout contract`
- 做了什么：
  - 新增 `src/models/optics/readout.py`，把 `I_out_full = |U_out_full|^2` 和 deterministic center crop 从调用脚本中抽离出来。
  - decoder 现在稳定返回 `U_out_full`、`I_out_full`、`I_out_roi`，不再把 full-grid 和 ROI 混在一起。
  - `output_crop_hw` 变成显式配置，crop 越界会报错。
- 为什么做：
  - Stage 3 freeze 明确要求 full-grid forward + ROI supervision，且 crop 不能隐藏在脚本中。
- 最小结论：
  - `scripts/check_optical_readout.py` 已在当前仓库状态下通过。
  - 当前重跑结果：
    - `I_out_full` shape: `(1, 1, 48, 48)`
    - `I_out_roi` shape: `(1, 1, 20, 20)`，改为 `(12, 12)` 时 ROI shape 正确变为 `(1, 1, 12, 12)`
    - oversized crop 会显式报错
    - 改变输入 phase 时，`full_delta = 1.748107e-03`，`roi_delta = 2.850121e-03`

## S3-04 Forward Sanity for L=1/3/5

- 对应任务：Issue 4 `feat: add Stage 3 forward sanity scripts for L=1/3/5`
- 做了什么：
  - 新增 `scripts/check_optical_forward_depths.py`，在一致的最小协议下检查 `L=1/3/5` 的 forward 行为。
  - 检查项覆盖：forward 成功、输出 key 完整、`U_out_full` 为 complex、强度非负、ROI shape 一致、不同 depth 输出不完全相同、invalid distance schedule 会显式失败。
- 为什么做：
  - Stage 3 在进入 fitting 前必须先证明 optics forward 本身可信，不依赖 electrical decoder 才能“看懂”输出。
- 最小结论：
  - `scripts/check_optical_forward_depths.py` 已在当前仓库状态下通过。
  - 当前重跑结果：
    - 三个 depth 都返回 `['I_out_full', 'I_out_roi', 'U0', 'U_out_full']`
    - `L=1/3/5` 的 full-grid shape 均为 `(1, 1, 48, 48)`，ROI shape 均为 `(1, 1, 20, 20)`
    - pairwise ROI delta:
      - `L1-L3 = 5.641614e-01`
      - `L1-L5 = 7.738391e-01`
      - `L3-L5 = 3.243636e-01`
    - 错误的 `inter_layer` 长度会被显式拒绝，而不是依赖旧的 layer-index special case

## S3-05 Decoder-Only Single-Sample Fitting

- 对应任务：Issue 5 `feat: add decoder-only single-sample fitting runner`
- 做了什么：
  - 新增 `scripts/train_decoder_only_single_sample.py`
  - 用 learnable input phase 直接驱动 optical decoder，不引入 encoder。
  - 在 ROI 上使用 normalized MAE 进行最小拟合，并保存 target / initial / best / final ROI、loss curve 和 summary。
- 为什么做：
  - Stage 3 需要先隔离 encoder 影响，验证 optical decoder stack 是否有最小可优化容量。
- 最小结论：
  - `outputs/optics/decoder_only_single_sample_smoke/summary.json` 记录了一个 `L=3` 的可复跑 smoke：
    - `initial_loss = 0.1463065892457962`
    - `best_loss = 0.03812457248568535`
    - `final_loss = 0.03893868252635002`
    - `loss_decrease = 0.10818201676011086`
    - `roi_change_mean_abs = 0.4759185314178467`
  - 当前结论仅说明 decoder-only 单样本路径可学，不代表论文设置或最终系统性能已复现。

## S3-06 Decoder-Only Small-Subset Depth Sweep

- 对应任务：Issue 6 `feat: run small-subset optical capacity experiment across L=1/3/5`
- 做了什么：
  - 先新增 `scripts/train_decoder_only_small_subset.py`，再补齐 `scripts/train_decoder_only_small_subset_sweep.py`
  - 对固定 deterministic subset、固定 ROI normalized MAE、固定步数预算和固定优化超参执行 `L=1/3/5` sweep
  - 为每个 depth 输出独立目录，并生成统一 `sweep_summary.json` 与 `depth_comparison.csv`
- 为什么做：
  - 单样本下降不足以支持 Issue 6 关闭；Stage 3 freeze 要求至少在小子集上比较 `L=1/3/5`
- 最小结论：
  - 来源：`outputs/optics/decoder_only_small_subset_sweep_run1/sweep_summary.json`
  - shared protocol：
    - `subset_size = 4`
    - `steps = 80`
    - `seed = 42`
    - `device = cpu`
    - `lr_phase = 0.15`
    - `lr_decoder = 0.03`
    - `freeze_decoder = false`
    - `loss = normalized_mae_roi`
  - fixed-protocol 结果：
    - `L=1`: `success_count = 4/4`, `mean_best_loss = 0.02864284673705697`, `mean_loss_decrease = 0.09716733871027827`
    - `L=3`: `success_count = 4/4`, `mean_best_loss = 0.03225723281502724`, `mean_loss_decrease = 0.09978684224188328`
    - `L=5`: `success_count = 4/4`, `mean_best_loss = 0.03135538613423705`, `mean_loss_decrease = 0.10351110668852925`
  - 当前结论仅说明三种 depth 在 tiny protocol 下都具备可优化容量，不构成论文性能排序结论。

## S3-07 Phase-Provider Hook

- 对应任务：Task3-7 `feat: add clean phase-provider hook for future Stage 4 integration`
- 做了什么：
  - 新增 `src/models/optics/phase_provider.py`
  - 明确 `PhaseProvider` callable contract：`upstream_input -> phase tensor`
  - 新增 `DirectPhaseProvider` 作为当前 Stage 3 默认 stub
  - 在 decoder 中稳定 `forward_from_phase(...)`、`forward_from_field(...)`，并补充 `forward_from_phase_provider(...)`
- 为什么做：
  - 让 optical core 与 future encoder 的实现方式解耦，同时不提前引入 Stage 4 trainer 或 joint model
- 最小结论：
  - 当前 hook 只负责接口边界，不负责任何训练或 encoder 实现
  - 已做的 smoke 结论：
    - `forward_from_phase(...)`、`forward_from_field(...)`、`forward_from_phase_provider(...)` 可运行
    - 现有 Issue 5/6 脚本在加入 hook 后仍能运行

## 当前 Stage 3 记录边界

- 已固定：
  - propagation primitive 的实现边界
  - full-grid forward + ROI supervision contract
  - `L=1/3/5` 的显式 layer semantics
  - decoder-only single-sample / small-subset 的最小验证路径
  - future Stage 4 的 phase-provider hook 入口
- 尚未在 Stage 3 中解决：
  - encoder / joint training
  - paper full-setting reproduction
  - hardware robustness / quantization / misalignment
  - dataset-scale end-to-end optical experiments
