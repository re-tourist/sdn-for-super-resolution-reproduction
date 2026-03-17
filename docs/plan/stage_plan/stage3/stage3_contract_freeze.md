# Stage 3 Contract Freeze

## 1. 文档定位

本文件用于在 Stage 3 实现开始前，冻结当前项目对 optical module verification 的核心共识。
它服务于后续 issue 拆解、接口设计和实现顺序安排，强调的是：

- 先冻结边界
- 再进入实现
- 不把 Stage 3 提前扩展成 Stage 4/5

本文件不是实现说明书，也不替代实验日志。Stage-3 immediate decisions 已经完成冻结，后续实现应同时遵循本文件与 `docs/integration/stage3_unresolved_params.md` 中记录的 finalized rules。

---

## 2. Stage 3 Goal

Stage 3 的正式目标是：

> 构建一个 paper-aligned optical decoder skeleton，并先完成 optical module 本身的可验证性确认。

本阶段优先支持以下三类能力：

1. forward sanity verification
2. decoder-only single-sample fitting
3. small-subset optical capacity validation

这意味着 Stage 3 当前追求的是：

- optical decoder 的正确输入输出 contract
- paper-aligned propagation / readout / crop 链路
- 在不依赖 electrical decoder 的前提下验证 optical capacity

这也意味着 Stage 3 当前不是：

- 最终 end-to-end super-resolution system
- 最终 joint training 方案
- 最终 paper full-setting reproduction

---

## 3. Stage 3 Scope

### 3.1 当前阶段要做什么

当前 Stage 3 的 in-scope 内容包括：

- optical propagation core
- phase-only input to coherent field 的主线适配
- output-plane intensity readout
- output ROI crop / FOV 对齐
- decoder-only verification mode
- `L=1/3/5` 的 forward sanity 和 optical capacity 初步验证

### 3.2 当前阶段不做什么

当前 Stage 3 的 out-of-scope 内容包括：

- electrical decoder migration
- upstream 全量 porting
- final encoder + optical decoder joint training
- final paper-setting hyperparameter alignment
- quantization-aware training
- misalignment / hardware robustness 扩展
- 自然图像重建 head、classification head 或其他任务层迁移

### 3.3 当前阶段的边界原则

Stage 3 的实现必须满足以下边界：

1. optical core 必须可以独立 forward 和验证，不依赖下游 electrical head 才能解释输出。
2. full propagation grid 与 output ROI 必须显式区分，不能把 crop 隐藏在脚本临时逻辑里。
3. `L=1/3/5` 的比较必须建立在明确 layer semantics 和 distance scheduling 之上。
4. 已冻结的 Stage-3 决策必须直接落到实现 contract 中，不能在实现时重新解释。

---

## 4. Optical Module Contract

### 4.1 最小接口链路

Stage 3 的 optical module contract 冻结为以下最小链路：

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi -> loss/metrics`

### 4.2 工程化职责拆分

| 环节 | 输入 | 输出 | 职责边界 |
| --- | --- | --- | --- |
| phase-to-field | `phi_lr` | `U0` | 将 phase-only LR pattern 解释为 coherent input field；主线默认振幅固定为 1 |
| optical propagation | `U0` | `U_out_full` | 执行 paper-aligned 多层 diffractive propagation；`L` 表示 trainable diffractive phase masks 数量，最终传播到 sensor plane 不计入 layer count；传播距离由显式 distance schedule 控制 |
| intensity readout | `U_out_full` | `I_out_full` | 计算输出平面的强度图 `|U|^2`，并保留 full-grid intensity 用于诊断、可视化和物理检查 |
| ROI crop | `I_out_full` | `I_out_roi` | 从 full propagation grid 中执行显式、确定性的 center crop；crop 逻辑不能隐藏在脚本中，配置层使用类似 `output_crop_hw` 的参数控制 |
| objective / metrics | `I_out_roi`, `target` | `loss`, `metrics` | 主监督对象固定为 `I_out_roi`；Stage 3 默认使用 normalized MAE，按单样本 ROI 计算 `sigma = sum(target) / sum(pred)`，分母加入 `epsilon`，batch 结果按样本求平均，并预留 PSNR / SSIM 评估 |

### 4.3 已冻结的 contract 决策

以下内容视为当前 Stage 3 已冻结的接口级共识：

1. 主线输入是 `phi_lr`，后续可来自 trainable phase tensor 或未来 encoder 输出。
2. `U_out_full` 和 `I_out_full` 是需要保留的中间结果，不能只返回 ROI。
3. 层数 `L` 的语义固定为 trainable diffractive phase masks 数量，且 `L ∈ {1, 3, 5}`。
4. 最终传播到 sensor plane 不计入 layer count；传播距离必须通过带显式语义的 distance schedule 表达，并区分 `input_to_first`、`inter_layer`、`last_to_sensor`。
5. ROI contract 固定为显式 center crop，且链路顺序固定为 `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi -> loss/metrics`。
6. 监督规则固定为 full-grid forward + ROI supervision：loss 和主要 metrics 只在 `I_out_roi` 上计算，`I_out_full` 保留用于诊断。
7. Stage 3 normalized MAE 的工程默认实现已经冻结：按单样本 ROI 计算 `sigma = sum(target) / sum(pred)`，显式加入 `epsilon` 数值保护，再对 batch 求均值。
8. optical module 只负责 direct optical readout，不包含 electrical decoder。

### 4.4 Distance schedule schema

Stage 3 固定采用 ordered list with explicit semantics 的距离表示，不再接受依赖 layer index 特判的隐式展开。

概念 schema 如下：

```yaml
distance_schedule:
  input_to_first: ...
  inter_layer: [...]
  last_to_sensor: ...
```

该 schema 的职责是：

- 明确输入面到第一层的传播距离
- 明确相邻可训练 diffractive phase masks 之间的传播距离
- 明确最后一层到 sensor plane 的传播距离

### 4.5 与后续文档的关系

Stage 3 已冻结的是 contract 结构与 immediate decisions 的工程口径，不是所有 paper 数值细节的最终对齐结果。
后续若要进入 paper-exact alignment，应在不改变 Stage 3 scope 的前提下继续参考 `docs/integration/stage3_unresolved_params.md` 中剩余的 P1 / P2 项。

---

## 5. Stage 3 Deliverables

本阶段的预期交付物冻结为：

1. `docs/plan/stage3_contract_freeze.md`
2. `docs/integration/stage3_unresolved_params.md`
3. optics core skeleton
4. intensity readout 与 ROI crop 实现
5. forward sanity script
6. decoder-only single-sample fitting mode
7. small-subset optical capacity validation 的最小运行壳

这些交付物的作用是让团队在不引入完整 end-to-end 系统的前提下，先判断 Stage 3 optical path 是否可信、是否可学、是否值得进入下一阶段。

---

## 6. Non-goals

以下内容明确不是当前 Stage 3 的目标，应后置到 Stage 4/5 或更后阶段：

1. electrical decoder migration 或 Pet-style reconstruction path
2. full upstream `onn` / `eoe` 系统移植
3. final joint training pipeline
4. 最终 paper-setting 对齐实验
5. phase quantization、misalignment vaccination、hardware material model
6. complex-valued input、amplitude-only input 等扩展主线
7. 最终 blind line-pair benchmark 自动化与系统化消融

---

## 7. 冻结说明

本文件是实现前的边界冻结说明。
在以下事项未完成前，不应把 Stage 3 直接推进到大规模移植或训练：

1. contract 边界确认
2. finalized Stage-3 immediate decisions 已被实现端明确接受
3. Stage 3 issue 顺序与验收边界确认

只有在这些前置条件稳定后，Stage 3 Issue 1 才算真正具备进入执行的条件。
