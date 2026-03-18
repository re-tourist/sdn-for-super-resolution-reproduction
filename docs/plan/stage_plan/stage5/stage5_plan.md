**GPT-5.4**

---

# 第五阶段具体规划（Stage 5 — 论文设定对齐）

---

# 1. 文档定位

本文档定义本仓库 **Stage 5（论文设定对齐）** 的工程规划与验收边界。

Stage 5 的任务不是做系统化消融，而是把论文主设定（数据、光学几何、encoder、loss、训练超参、评估协议）落实成 **可运行、可复盘、可追踪** 的工程协议。

系统化消融、量化 sweep、鲁棒性与大规模对比实验属于 Stage 6。

参考优先级（与现有文档一致）：

1. `docs/plan/plan_overview.md`
2. `docs/plan/stage3_contract_freeze.md`
3. `docs/plan/stage_plan/stage4/stage4_protocol_freeze.md`
4. 当前 `src/` 与 `scripts/`
5. `docs/paper/paper_notes.md`
6. `docs/execution/results_summary.md`
7. `docs/execution/experiment_log.md`

范围冲突处理规则：

- 当 `docs/paper/paper_notes.md` 记录的是更宽的复现总范围，而本文件记录的是当前阶段排期时，以本文件为准。
- 特别是 quantization、misalignment、systematic ablations 虽属于总复现范围，但当前明确后置到 Stage 6。

---

# 2. Stage 5 核心目标

1. **数据协议对齐**：实现论文中的 96×96 EMNIST display 数据构造与 train/val/test 划分。
2. **光学几何对齐**：实现 paper-aligned 的 grid 尺寸、zero padding、距离 schedule 与初始化策略。
3. **encoder 对齐**：实现论文主线的 phase-only encoder 输出与 `phi_lr` contract。
4. **损失与训练超参对齐**：实现 normalized MAE + efficiency term，匹配 Adam + LR + batch + epoch 设定。
5. **评估协议对齐**：落实 PSNR / SSIM 口径、bicubic baseline、blind line-pair test。
6. **可复现工程化**：用 configs 与日志把上述设置固化，输出可追踪工件与文档。

---

# 3. Stage 5 非目标

以下内容 **不属于 Stage 5**，应明确后置到 Stage 6：

- 系统化 L=1/3/5 sweep 与完整消融矩阵
- efficiency penalty 的系统化 ablation
- quantization sweep（16/8/6/4/2 bit）
- misalignment / robustness / hardware-aware 训练
- 复杂调制（complex-valued / amplitude-only）系统化对比
- 大规模参数调参或框架级重构

---

# 4. 进入 Stage 5 的前置条件

进入 Stage 5 前，需满足 Stage 4 通过条件（见 `docs/plan/stage_plan/stage4/stage4_protocol_freeze.md`）：

- 单样本 overfit 闭环通过
- 小子集闭环训练通过
- encoder / optics 梯度可观测且非零

当前仓库已存在：

- `docs/execution/stage4_single_sample_report.md`（已通过）
- `docs/execution/stage4_small_subset_report.md`（已通过）

---

# 5. 当前工程状态与 Stage 5 缺口

## 已有基础

- Stage 3 optical contract 已冻结：`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- Stage 4 已建立最小闭环训练路径（非论文设定）
- PSNR / SSIM 评估工具与电子 baseline 已通过 Stage 1/2 验证

## Stage 5 缺口

- paper-aligned dataset（96×96 display）
- paper-aligned optics config（200/400 grid + distances）
- paper-aligned encoder（phase-only）
- paper loss + efficiency term
- paper-aligned trainer 与 config
- paper-aligned eval + bicubic baseline + line-pair test

---

# 6. 论文设定对齐目标（paper-aligned v1）

## 6.1 数据协议

- 数据源：EMNIST letters
- 原始 28×28 → bicubic 插值到 32×32
- 96×96 display 由 32×32 patch 组合构成
- 数据量：train 60,000 / val 6,000 / test 6,000
- 内容分布：train/val 含 1~4 个字母；test 含 6~9 个字母
- 增强：随机旋转（0/90/180/270）、随机翻转、随机对比度

## 6.2 光学几何与网格

- diffractive neuron sampling：0.533 λ
- layer size：200×200 pixels
- input/output FOV：96×96 pixels
- zero padding：400×400
- depth：L = 1 / 3 / 5
- 初始化：phase masks = 0

距离（来自 `paper_notes.md`，需显式映射到代码的 distance schedule）：

- L=1：d1 = 6.667λ，d3 = 173.333λ
- L=3：d1 = 4λ，d2 = 53.334λ，d3 = 53.334λ
- L=5：d1 = 2.667λ，d2 = 66.667λ，d3 = 80λ

> 注：论文未明确说明这些距离与 `input_to_first / inter_layer / last_to_sensor` 的一一对应，需要在 Stage 5 protocol 中明确映射。

## 6.3 Encoder 与相位约束

- 输入：96×96 HR target image
- 输出：低分辨率 phase-only pattern `phi_lr`
- `phi_lr` 空间尺寸预期为 32×32（与 3× SR factor 对齐），但需在 Stage 5 协议中显式确认
- 相位范围映射需显式（例如 `[0, 2π)` 或 `[-π, π]`），不能隐式藏在 optical core 中

## 6.4 Loss 与效率项

主损失：normalized MAE

\[
\mathcal L = \frac{1}{N}\sum_i |y_i - \sigma \hat y_i| + \gamma e^{-\eta}
\]

其中：

\[
\sigma = \frac{\sum_i y_i}{\sum_i \hat y_i + \epsilon}
\]

\[
\eta = 100 \times \frac{P_o}{P_i}
\]

论文建议：

- γ = 0.005 for L=1
- γ = 0.015 for L=3
- 其它情况 γ = 0（需明确是否对 L=5 也为 0）

Stage 5 的实现要求：

- loss 必须支持 configurable gamma，而不是把 L=5 gamma 静态拍死在代码里
- 若实现阶段暂未拍板 L=5 gamma，则必须把默认值与理由写入 config / summary / protocol freeze

## 6.5 训练设定

- optimizer：Adam
- 学习率：decoder 0.001 / encoder 0.0005
- batch size：40
- epochs：500
- 训练需支持 encoder / decoder 参数分组

## 6.6 评估协议

- PSNR / SSIM
- bicubic baseline（需 anti-aliasing）
- blind line-pair / resolution target 测试
- train/val/test 口径与输出 crop 对齐必须在 config 中显式记录

## 6.7 工程化输出

- configs：可复现、可 diff
- summary.json / metrics.json / checkpoints / sample previews
- 运行命令与 config snapshot 必须落地

---

# 7. 待确认与假设清单（Stage 5 必须显式记录）

| 未决项 | 当前默认处理 | 主要拍板 issue | 当前不允许发生的事 |
| --- | --- | --- | --- |
| `phi_lr` 的精确尺寸是否固定为 32×32 | 默认假设 `96 / 3 = 32`，但不得伪装成论文唯一明示 | Issue 5.1, 5.4 | 在 encoder 实现里把 `32×32` 写成“论文已明确唯一正确值” |
| 96×96 display 的 tile 规则 | 需要显式写进 dataset config 与预览 | Issue 5.1, 5.2 | 在 dataset 代码里隐式写死随机布局而不记录 |
| distance schedule 与 `input_to_first / inter_layer / last_to_sensor` 的映射 | 必须显式写进 optics config 与协议文档 | Issue 5.1, 5.3 | 在 optics 实现中靠隐式位置约定或 layer-index special case 拍板 |
| efficiency penalty 是否对 L=5 也为 0 | 必须做成 configurable default，并在文档中记录 | Issue 5.1, 5.5 | 在 loss 代码里静态宣布 L=5 gamma 已被论文唯一确定 |
| phase mapping 使用 `[0, 2π)` 还是 `[-π, π]` | 允许实现 configurable mapping | Issue 5.1, 5.4 | 把某个 mapping 写成论文唯一明确事实 |
| σ 的归一化按逐样本还是逐 batch | 必须在 loss 实现与 protocol freeze 中明确 | Issue 5.1, 5.5 | 在训练代码里默认拍板且不记录 |
| output FOV / full-grid crop 对齐是否有额外 margin | 必须在 eval / config 中显式记录 | Issue 5.1, 5.3, 5.7 | 在 eval 里悄悄改 crop 口径 |

这些条目必须被写入 Stage 5 protocol freeze 文档，不能隐性处理。

---

# 8. Stage 5 任务拆解（建议）

- **Stage 5A — 协议冻结**：写出 paper-aligned protocol freeze，并冻结未决项处理方式。
- **Stage 5B — 数据协议落地**：实现 96×96 EMNIST display 数据构造与增强。
- **Stage 5C — 光学配置对齐**：加入 paper grid/padding/distances 的 optics config（L=1/3/5）。
- **Stage 5D — Encoder 对齐**：实现 paper-aligned phase-only encoder。
- **Stage 5E — 损失函数对齐**：normalized MAE + efficiency penalty 可配置化。
- **Stage 5F — 训练骨架对齐**：Stage 5 trainer + configs，支持长训与 resume。
- **Stage 5G — 常规评估协议对齐**：PSNR/SSIM + bicubic baseline。
- **Stage 5H — blind line-pair 协议对齐**：line-pair / resolution target 生成与评估挂接。
- **Stage 5I — paper-aligned smoke**：短跑验证 pipeline 稳定。
- **Stage 5J — paper-aligned main run**：主线训练与评估结果记录。
- **Stage 5K — 文档收口**：更新执行日志、结果摘要，给出 Stage 6 入口结论。

---

# 9. Stage 5 验收标准

## 9.1 最小工程完成标准

1. 96×96 display 数据集可稳定生成，预览样本可解释。
2. paper-aligned optics config 能在 L=1/3/5 下 forward 成功且 shape 正确。
3. paper-aligned loss / efficiency term 可开关，并在 config 中可追溯。
4. paper-aligned trainer 可完成 smoke 训练且无 NaN/Inf。
5. 常规 eval runner 可输出 PSNR / SSIM，并可复用 bicubic baseline。
6. blind line-pair 测试可独立生成并挂接到 eval 路径。
7. main-run config、启动命令和工件协议已定义清楚，可作为 launch-ready recipe 使用。
8. 所有关键假设已在 protocol freeze 与 summary 中明确记录。

## 9.2 更强的复现完成标准（资源允许时）

1. 至少完成一次 paper-aligned 主线训练（建议 L=5 phase-only）。
2. main run 的 val/test 结果可以与 bicubic baseline 和 blind line-pair 测试一起复盘。
3. main run 中所有资源相关妥协都被显式记录，而不是隐含在代码里。

## 9.3 对 Prompt 的含义

- 后续给 Codex 的 implementation prompt 应优先对齐 `9.1 最小工程完成标准`。
- `9.2` 属于更强证据目标，不能反向逼迫前置 issue 提前扩张成“大而全框架”。
- 若主线训练受算力或时间预算限制，报告必须把“代码/配置已就绪”和“长训尚未跑完”明确分开记录。

---

# 10. 风险与排错优先级

1. **显存/内存压力**：400×400 propagation grid 显著增加显存。
   - 方案：先跑 CPU/小 batch smoke，再转 GPU；必要时引入 gradient checkpointing，但需明确记录。
2. **数据协议偏差**：96×96 tile 规则若偏差，将导致结果不可对齐。
   - 方案：保存可视化样本并在文档中记录 tile 规则。
3. **距离映射歧义**：论文距离参数映射不清会导致物理设置偏差。
   - 方案：在 protocol freeze 中显式写出 mapping 与理由。
4. **训练不稳定**：长训 + efficiency term 可能引入数值不稳。
   - 方案：先关闭 efficiency term 做 smoke；再按 L=1/3 的 γ 值逐步启用。

---

# 11. 文档同步建议

完成 Stage 5 后应更新：

- `docs/execution/experiment_log.md`
- `docs/execution/results_summary.md`
- `docs/plan/reproduction_plan.md`
- `docs/paper/paper_notes.md`（补充明确化的未决参数）

---

# 12. Stage 6 入口条件（简述）

进入 Stage 6 前，至少需要满足：

1. `9.1 最小工程完成标准` 已全部满足。
2. 已有至少一次 paper-aligned 主线训练被实际启动并形成可复盘记录。
3. 若主线训练尚未达到理想预算，也必须把资源限制、当前结果和阻塞点写清楚。

换句话说，Stage 6 的前提不是“Stage 5 所有长训都已经完美跑满”，而是：

- paper-aligned pipeline 已落地
- 主线运行路径已被真实执行过
- 后续系统化实验有可信起点可继承
