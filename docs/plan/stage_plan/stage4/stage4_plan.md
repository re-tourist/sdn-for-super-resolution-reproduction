**GPT-5.3**

---

# 第四阶段具体规划（Stage 4 — 最小闭环训练）

---

# 1. 文档定位

本文档定义本仓库 **Stage 4（最小闭环训练）** 的工程实现方案。

该阶段的目标不是复现论文指标，而是解决一个更基础、更关键的问题：

> **当电子编码器（encoder）接入已经实现的光学解码器（diffractive decoder）后，整个系统是否具备在梯度驱动下的可学习性。**

Stage 4 必须：

* 承接 **Stage 3 已冻结的 optical contract**
* 复用已有 optical implementation
* 复用 Stage 1 / 2 已验证的训练与评估组件
* 在 **小规模、可调试、可解释** 的设置下验证 end-to-end learnability

Stage 4 **不是**：

* 论文完整设定复现
* 大规模训练
* 参数 sweep 或消融实验

这些工作属于 **Stage 5 / Stage 6**。

---

# 2. Stage 4 目标

Stage 4 的唯一核心问题：

> **encoder + optical decoder 组成的系统能否在端到端训练下学习。**

具体工程目标：

1. **接入最小 encoder**

2. **复用 Stage 3 optical decoder**

3. **构建最小闭环训练 pipeline**

4. **完成两个 learnability sanity check**

   * 单样本 overfit
   * 小数据集训练

5. **保存足够多的中间产物用于调试**

   * encoder 输出 phase pattern
   * optical ROI intensity
   * loss curve
   * gradient statistics

6. 确认以下现象：

* loss 能稳定下降
* encoder 参数收到梯度
* optical 层参数收到梯度
* 输出 ROI 逐渐接近目标图像

---

# 3. Stage 4 非目标

以下工作 **明确不属于 Stage 4**：

### 论文设定对齐

* 96×96 完整数据 protocol
* 完整数据量
* 500 epoch 训练
* 论文 exact optical grid

### 系统消融

* L = 1 / 3 / 5 sweep
* efficiency penalty ablation
* complex vs phase modulation
* quantization study

### 工程扩展

* blind line-pair test
* misalignment robustness
* hardware-aware simulation

这些将属于 **Stage 5 / Stage 6**。

---

# 4. 当前项目现状与关键约束

本节总结当前 repo 的 **真实工程状态**。

---

# 4.1 已经存在的可复用资产

### Repo 基础结构

项目已经完成：

* Stage 0 repo scaffold
* planning docs
* execution logging
* evaluation helpers

这些已经记录在 `PROJECT_CONTEXT.md` 中。 

---

### 数据与评估路径（Stage 1 / 2）

已经验证：

* EMNIST 数据处理路径
* PSNR / SSIM 评估工具
* 数据 sanity checks
* interpolation baseline
* pure electronic baseline

电子 baseline 已经完成基本 sanity check。 

这意味着：

**Stage 4 不需要重新设计数据与评估系统。**

---

### 光学模块（Stage 3）

当前 repo 已实现：

* diffractive optical decoder
* propagation module
* phase utilities
* optical readout

并且已经完成以下验证：

* optical forward sanity
* decoder-only single sample fitting
* decoder-only small subset fitting

这些验证通过 **toy synthetic target** 完成。

当前 optical decoder forward path 已定义为：

```
phi_lr
  ↓
U0
  ↓
U_out_full
  ↓
I_out_full
  ↓
I_out_roi
```

该 contract 在 Stage 3 已冻结。

---

### Stage 3 capacity verification scripts

当前存在以下 Stage 3 验证脚本：

```
scripts/train_decoder_only_single_sample.py
scripts/train_decoder_only_small_subset.py
scripts/train_decoder_only_small_subset_sweep.py
```

这些脚本的关键特征：

* synthetic target
* toy grid
* decoder-only fitting

例如 `train_decoder_only_small_subset.py` 使用：

```
input_pattern_hw = (24,24)
layer_hw = (32,32)
propagation_hw = (48,48)
output_crop_hw = (20,20)
```

并通过 synthetic target library 拟合 ROI。 

因此：

> Stage 3 的 optical 验证 **仍然是 synthetic / toy setup**。

---

### 电子 baseline trainer

当前 repo 有可信训练骨架：

```
scripts/train_electronic_baseline.py
```

该脚本：

* 已用于 Stage 1 / 2
* 可作为 Stage 4 trainer 参考结构

---

# 4.2 当前仍然缺失的基础设施

根据 PROJECT_CONTEXT：

当前 repo **尚未实现**：

* 通用 Stage 4 trainer
* Stage 4 evaluation runner
* dataset module
* config system
* automated tests

具体情况：

```
scripts/train.py      -> empty
scripts/eval.py       -> empty
scripts/visualize.py  -> empty
```

因此：

> **当前 repo 还没有真正的 Stage 4 training pipeline。**

这将是 Stage 4 的主要工作。

---

# 4.3 当前哪些结论是可信的

以下结论已经被验证：

### Optical forward path

光学传播实现：

* shape 正确
* forward 稳定
* intensity ROI 可读

### Decoder capacity

在 synthetic target 上：

* loss 可以下降
* output ROI 可变化
* phase mask 可更新

这说明：

> **optical decoder 本身具备表达能力。**

---

# 4.4 当前哪些结论仍是初步验证

以下结论 **尚未被证明**：

### End-to-end learnability

目前没有证据表明：

```
encoder → optical decoder
```

组合后系统能学习。

### Encoder compatibility

目前没有验证：

* encoder 输出 phase pattern 是否可训练
* encoder 梯度是否稳定

### Dataset compatibility

Stage 3 训练使用 synthetic targets。

尚未验证：

```
真实 EMNIST 图像
→ optical decoder
```

是否可训练。

---

# 5. Stage 4 最小闭环系统定义

Stage 4 系统结构：

```
HR image
   ↓
minimal encoder
   ↓
phi_lr
   ↓
optical decoder (Stage 3)
   ↓
I_out_roi
   ↓
loss
```

---

## 5.1 输入

HR target image

建议：

```
32×32 或 48×48
```

（非论文最终尺寸）

---

## 5.2 encoder 输出

```
phi_lr
shape = (1, H_lr, W_lr)
```

并满足：

```
phi_lr ∈ [-π, π]
```

---

## 5.3 optical decoder

完全复用 Stage 3 实现：

```
DiffractiveDecoder
```

不允许重新设计光学 core。

---

## 5.4 loss

Stage 4 可使用简化 loss：

```
normalized MAE
```

Stage 3 已实现类似 loss。

---

## 5.5 输出

```
I_out_roi
```

与 target image 对齐。

---

# 6. 关键设计决策

---

## 6.1 encoder 最小实现

Stage 4 不追求论文 encoder。

建议使用：

```
3-layer CNN
```

结构：

```
Conv → ReLU
Conv → ReLU
Conv → tanh
```

输出：

```
phi_lr
```

---

## 6.2 phase mapping

使用：

```
phi = π * tanh(x)
```

理由：

* 与 Stage 3 phase constraint 一致
* 梯度稳定

---

## 6.3 optical grid

Stage 4 保持：

```
24 / 32 / 48 / 20
```

toy grid。

原因：

* 易调试
* 计算成本低

---

## 6.4 训练规模

Stage 4 训练规模：

```
dataset size = 8~32
batch size = 1~4
epoch = 50
```

---

# 7. 开发任务拆解

Stage 4 拆分为：

```
Stage 4A
Stage 4B
Stage 4C
Stage 4D
```

---

# Stage 4A — Encoder 接入

## 目标

实现最小 encoder。

---

## 输入

HR image

---

## 输出

```
phi_lr
```

---

## 产物

```
src/models/encoder/minimal_encoder.py
```

---

## 验收标准

* 输出 shape 正确
* phase range 正确
* forward 无 NaN

---

## 风险

encoder 输出范围不稳定。

---

# Stage 4B — 构建 Stage 4 trainer

## 目标

实现通用训练脚本：

```
scripts/train_stage4_minimal.py
```

---

## 输入

dataset + config

---

## 产物

Stage 4 trainer

---

## 功能

trainer 必须支持：

* optimizer
* artifact saving
* gradient check
* logging

---

## 验收标准

训练能运行：

```
10 steps
```

---

## 风险

memory explosion。

---

# Stage 4C — 单样本 overfit

## 目标

验证系统 learnability。

---

## 输入

single image

---

## 产物

```
outputs/stage4_single_sample
```

---

## 验收标准

```
loss ↓
output 接近 target
```

---

## 风险

梯度无法传到 encoder。

---

# Stage 4D — 小数据集训练

## 目标

验证训练稳定性。

---

## 输入

```
dataset size = 16
```

---

## 产物

```
outputs/stage4_small_dataset
```

---

## 验收标准

loss 稳定下降。

---

# 8. 验证与验收标准

Stage 4 必须通过以下 sanity tests。

---

## 单样本 overfit

要求：

```
loss ↓
ROI 变化
```

---

## 小数据集训练

要求：

```
loss ↓
```

---

## 梯度检查

必须确认：

```
encoder gradients ≠ 0
optical gradients ≠ 0
```

---

## 中间产物保存

必须保存：

```
encoder output phase
optical ROI
loss curve
```

---

# 9. 风险与排错优先级

最可能问题：

---

## 1 encoder gradient 消失

排查：

* phase mapping
* tanh saturation

---

## 2 optical loss 不下降

排查：

* ROI crop
* intensity scaling

---

## 3 output 无结构

排查：

* propagation grid
* phase init

---

# 10. 文档同步建议

Stage 4 完成后需更新：

```
docs/execution/experiment_log.md
docs/execution/results_summary.md
PROJECT_CONTEXT.md
```

---

# 11. 进入 Stage 5 前的出关条件

必须满足：

1️⃣ 单样本 overfit 成功

2️⃣ 小数据集训练稳定

3️⃣ encoder / optical gradients 正常

4️⃣ 输出 ROI 有结构

5️⃣ trainer pipeline 稳定

---

满足以上条件后：

项目进入 **Stage 5：论文设定对齐阶段**。
