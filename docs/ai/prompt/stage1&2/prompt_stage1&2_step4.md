TASK:
Implement Stage1&2-4: a pure electronic baseline aligned with the final optical pipeline.

GOAL:
Build a bottleneck-constrained all-electronic encoder-decoder baseline.
This is NOT a generic super-resolution network.
It must model the same task semantics as the target optical pipeline:

HR image -> electronic encoder -> low-resolution latent -> electronic decoder -> reconstructed HR image

The purpose is:
1. provide a fair pure electronic reference
2. verify learnability under the same bottleneck
3. serve as a comparison point for the future optical decoder

PROJECT CONTEXT:
- Current project is still in Stage1/2 groundwork
- Unified evaluation pipeline already exists
- Interpolation baseline already exists
- Do NOT expand scope into optical modules, paper-aligned ablations, or system refactors

TARGET FILES:
1. src/models/electronic_baseline.py
2. scripts/train_electronic_baseline.py

ALLOWED CHANGES:
- Add the two files above
- Add very small local helpers only if absolutely necessary

FORBIDDEN CHANGES:
- Do NOT modify optical modules
- Do NOT modify interpolation baseline script
- Do NOT modify evaluation pipeline implementation
- Do NOT rewrite dataset system globally
- Do NOT introduce large framework refactors
- Do NOT add U-Net / EDSR / RCAN / GAN / transformer / diffusion / perceptual loss
- Do NOT add skip connections that bypass the bottleneck

MODEL REQUIREMENTS:
Implement a lightweight bottleneck autoencoder.

Default configuration:
- input shape: [B,1,96,96]
- hr_size = 96
- sr_factor = 4
- latent spatial size = 24x24
- latent_channels = 1
- output shape: [B,1,96,96]

Required semantics:
- explicit bottleneck must exist
- bottleneck must be inspectable
- no high-resolution skip path from encoder to decoder
- output range must be in [0,1]

Recommended structure:
Encoder:
- conv block
- downsample 96 -> 48
- conv block
- downsample 48 -> 24
- conv projection -> latent [B,1,24,24]

Decoder:
- conv block
- upsample 24 -> 48
- conv block
- upsample 48 -> 96
- output conv -> 1 channel
- sigmoid or equivalent explicit [0,1] output mapping

TRAINING SCRIPT REQUIREMENTS:
Implement a minimal runnable training script.

Training semantics:
HR -> encoder -> LR latent -> decoder -> reconstructed HR

Loss:
- use L1 / MAE only
- do NOT add optical efficiency penalty
- do NOT add quantization loss
- do NOT add phase regularization
- do NOT add adversarial/perceptual losses

Evaluation:
Must reuse existing unified evaluation entry:
from src.eval.evaluator import evaluate_batch

Do NOT reimplement PSNR/SSIM.

Training outputs:
- train loss
- val PSNR
- val SSIM
- metrics.json
- summary.json
- sample reconstruction images
- best checkpoint
- final checkpoint

Suggested output dir:
outputs/electronic_baseline/<run_name>/

DATA REQUIREMENTS:
- Prefer current repo’s existing Stage1/2 dataset shell
- If no formal dataset pipeline exists yet, use the same minimal data construction style as current interpolation baseline
- Do NOT refactor the whole dataset system for this task
- In your final response, explicitly state what data path / construction logic you used

DEFAULT TRAINING CONFIG:
- epochs = 10
- batch_size = 32
- lr = 1e-3
- optimizer = Adam

CLI REQUIREMENTS:
Support at least:
- --output-dir
- --epochs
- --batch-size
- --lr
- --seed
- --hr-size
- --sr-factor
- --latent-channels
- --num-samples
- --download
- --save-samples

ACCEPTANCE CRITERIA:
Task is complete only if ALL conditions below are satisfied:

1. A trainable pure electronic baseline model is added
2. The model has an explicit low-resolution bottleneck
3. Default latent is low-resolution single-channel representation
4. No high-resolution skip path bypasses the bottleneck
5. Training script runs independently
6. Training script outputs reconstruction samples, loss, PSNR, SSIM
7. Existing evaluate_batch(...) is reused
8. Output files include metrics.json and summary.json
9. Scope remains local and small
10. Final response clearly explains model structure, data assumptions, run commands, and known limitations

SMOKE TEST:
Provide one smoke-test command and one normal training command.

FINAL RESPONSE FORMAT:
1. Task understanding
2. Files added/modified
3. Model structure
4. Training/data assumptions
5. Run commands
6. Output files
7. Risks / known limitations
8. Completion status against acceptance criteria

**GPT-5.4 Thinking**

我先把核心判断说清楚，再给你一份可直接交给 Codex 的提示词。

## 为什么这个电子 baseline 应该这样设计

这篇论文的主线并不是“给一个普通 LR 图像做超分”，而是：**把 96×96 的目标图像先压缩成低分辨率 modulation pattern，再由后端 decoder 重建回高分辨率输出**。也就是说，真正要对齐的对象是“**受限信息瓶颈下的重建能力**”，而不是普通意义上的超分网络。论文里前端是 CNN 编码器，后端是全光衍射解码器；目标数据在数值实验中是 96×96 的 EMNIST 拼图，评估指标用 PSNR / SSIM。

所以，一个**合适且公平**的 pure electronic baseline，不应该一上来就用很强的 U-Net 或带大量 skip connection 的大网络，否则它比较的就不是“同一瓶颈下电子 vs 光学”的差异，而是“强电子网络 vs 受限光学系统”的差异。更合理的做法是：

* 保留与最终光学主线一致的**低分辨率瓶颈**
* 做成一个**全电子 encoder-decoder**
* 输入仍是 HR target
* 中间 latent 的空间分辨率对齐 SLM / LR pattern
* 输出重建回 HR 图像
* 用统一评估 pipeline 做 PSNR / SSIM

这和论文的系统结构更一致：论文前端 encoder 本身也是 CNN，并把高分辨率图像编码为低分辨率表示；后端 decoder 再恢复输出。论文原始 encoder 甚至包含 4 个 convolutional blocks、flatten、fully connected 和 unflatten 结构。

基于你当前 Stage1/2 的目标，我建议这一步的电子 baseline 采用：

> **bottleneck-constrained all-electronic autoencoder baseline**

而不是：

> 普通 SRCNN / UNet 风格“从外部 LR 输入恢复 HR”的 baseline

因为前者才真正贴合你后面要做的光学主线。

---

下面是我整理好的 **Codex 提示词**。

---

# Task1&2-4 AI 编码提示词

（纯电子 baseline：与最终光学主线对齐的 bottleneck-constrained baseline）

你可以把下面整段直接交给 Codex。

---

# TASK

Stage1&2-4：实现一个与最终光学主线对齐的 pure electronic baseline

---

# 背景

当前项目正在复现论文：

**Super-resolution image display using diffractive decoders**

该论文的核心系统不是普通的图像超分网络，而是：

* 一个 **electronic encoder**
* 一个 **all-optical diffractive decoder**
* 二者联合训练

其中，电子编码器接收 **高分辨率目标图像**，输出 **低分辨率 modulation pattern**；后端 decoder 再根据这个低分辨率表示重建出高分辨率输出图像。该系统本质上更接近一个“**受限 bottleneck 下的 encoder-decoder / autoencoder**”，而不是普通“给定 LR 输入恢复 HR”的 SR 网络。论文数值实验使用 96×96 的 EMNIST display dataset，评估指标主要是 PSNR / SSIM。

当前项目仍处于 **Stage 1 / Stage 2 groundwork**，已经完成或正在推进：

* dataset sanity check
* unified evaluation pipeline
* interpolation baseline

现在需要实现一个 **pure electronic baseline**，其目标不是追求极致性能，而是：

1. 给出一个**与最终光学主线结构更公平**的电子参考
2. 验证当前任务在相同 bottleneck 下是否可学
3. 为后续 optical model 提供对照基线

---

# 本任务的关键设计原则

请特别注意：

这个电子 baseline **不要**实现成一个“很强的大型普通 SR 网络”（例如大 U-Net、EDSR、带大量 skip 的强重建器），因为那会破坏与最终光学系统的公平性。

本任务应实现的是：

> **一个受限 bottleneck 的全电子 encoder-decoder baseline**

也就是说：

* 输入：HR target image
* 中间：LR latent representation（空间尺寸对齐最终光学系统的 LR pattern）
* 输出：reconstructed HR image

这个 latent 需要模拟最终光学系统中的低分辨率 modulation pattern，而不是允许网络直接用高维隐藏特征绕过瓶颈。

---

# 任务目标

实现一个最小可用、低耦合、可训练、可评估的 **pure electronic bottleneck baseline**。

建议目标结构：

```text
HR image (96x96, 1ch)
    -> electronic encoder
    -> LR latent (default: 24x24, 1ch for k=4)
    -> electronic decoder
    -> reconstructed HR image (96x96, 1ch)
```

其中：

* 默认 `hr_size = 96`
* 默认 `sr_factor = 4`
* 默认 latent spatial size = `96 / 4 = 24`
* 默认 latent channel = `1`

这个设计是为了尽量贴合论文 phase-only LR representation 的主线设定：论文主线强调由 encoder 生成低分辨率 phase-only representation，再由 decoder 重建输出。

---

# 你需要完成的内容

请实现以下最小模块：

```text
src/models/electronic_baseline.py
scripts/train_electronic_baseline.py
```

如果确实需要极小的辅助函数，可以新增很小的本地工具，但不要顺手重构整个项目。

---

# 1. 模型实现要求

请在：

```text
src/models/electronic_baseline.py
```

中实现一个轻量级全电子 encoder-decoder。

---

## 1.1 推荐模型结构

请优先实现一个**简洁、受限、容易训练**的结构，例如：

### Encoder

将 `96x96` 输入压缩到 `24x24` latent：

* Conv block
* Downsample 96 -> 48
* Conv block
* Downsample 48 -> 24
* Conv projection -> latent channel = 1

### Decoder

将 `24x24` latent 重建回 `96x96`：

* Conv block
* Upsample 24 -> 48
* Conv block
* Upsample 48 -> 96
* Output conv -> 1 channel

你可以使用：

* strided convolution
* bilinear upsample + conv

但不要引入过重架构。

---

## 1.2 必须满足的结构约束

### 必须有明确 bottleneck

不要让模型变成“高维 feature 通过 skip connection 直接泄露到输出”。

因此：

* **不要使用 U-Net 式 encoder-decoder skip connections**
* 不要在 encoder 与 decoder 之间保留高分辨率特征捷径
* bottleneck 必须显式存在且可检查

---

## 1.3 latent 设计

默认实现：

```text
latent shape = [B, 1, 24, 24]
```

原因：

* 与 phase-only optical pipeline 更接近
* 对齐“低 SBP wavefront modulator”思想
* 更适合作为公平电子 baseline

同时，请把以下参数做成可配置：

```text
latent_channels
hr_size
sr_factor
```

默认：

```text
latent_channels = 1
hr_size = 96
sr_factor = 4
```

---

## 1.4 输出范围

输出图像应映射到 `[0,1]`，以便与当前统一评估 pipeline 保持一致。

建议：

* 最后一层使用 `sigmoid`
* 或者在 forward 末尾显式 clamp / normalize 到 `[0,1]`

但要保持行为清楚，不要隐式做太多魔法。

---

# 2. 训练脚本要求

请在：

```text
scripts/train_electronic_baseline.py
```

中实现一个最小可运行训练脚本。

---

## 2.1 数据逻辑

请优先复用当前 Stage1/2 已经形成的 96×96 数据壳。

当前论文数值实验的数据构造是：

* EMNIST letters 原始 28×28
* 先 bicubic 到 32×32
* 再 tiled 成 96×96 图像
* train / val / test 分别构造不同数量的手写字母组合图像

但本任务属于 Stage1/2 基础阶段，因此要求是：

* **优先复用当前 repo 已有的数据构造方式**
* 如果 repo 里还没有完整正式 dataset pipeline，可采用与当前 interpolation baseline 一致的最小独立实现
* 不要为了这个任务去重构整个 dataset 系统

请在反馈中明确说明你采用的数据来源与数据构造方式。

---

## 2.2 训练目标

训练目标不是“输入外部 LR，恢复 HR”，而是：

```text
HR -> encoder -> LR latent -> decoder -> reconstructed HR
```

也就是一个**受限信息瓶颈下的重建任务**。

---

## 2.3 loss

第一版请使用最简单、最稳定的主损失：

```text
L1 / MAE reconstruction loss
```

不要在这一版中引入：

* optical efficiency penalty
* phase regularization
* quantization loss
* adversarial loss
* perceptual loss

理由：

这些属于后续 optical / paper-aligned 阶段变量，不适合现在混入 baseline。

论文的正式 joint loss 包含归一化 MAE 和效率项，但效率项主要与光学输出能量集中有关，对纯电子 baseline 不应直接照搬。

---

## 2.4 评估

必须复用已有统一评估入口：

```python
from src.eval.evaluator import evaluate_batch
```

不要重写 PSNR / SSIM。

训练过程中至少记录：

* train loss
* val PSNR
* val SSIM

并在训练结束后导出：

```text
metrics.json
summary.json
```

---

## 2.5 checkpoint / outputs

建议输出目录结构：

```text
outputs/electronic_baseline/
  checkpoints/
  samples/
  train_curve.png
  metrics.json
  summary.json
```

至少保存：

* best checkpoint
* final checkpoint
* 若干 sample reconstruction 图
* metrics summary

---

# 3. 最小训练配置建议

为了保证 Stage1/2 可快速验证，请提供默认的轻量训练配置，例如：

```text
epochs = 10
batch_size = 32
lr = 1e-3
optimizer = Adam
```

并支持命令行参数覆盖。

如果方便，也请支持一个更小的 smoke test 配置，例如：

```text
--num-samples 128
--epochs 2
```

用于快速黑盒验收。

---

# 4. CLI 参数建议

建议训练脚本至少支持：

```text
--split train
--seed 42
--output-dir outputs/electronic_baseline/run1
--epochs 10
--batch-size 32
--lr 1e-3
--hr-size 96
--sr-factor 4
--latent-channels 1
--num-samples 1000
--save-samples
--download
```

如果存在 val/test 评估逻辑，也可额外支持：

```text
--eval-split val|test
```

---

# 5. 设计原则（非常重要）

## 5.1 最小改动原则

不要修改：

```text
optical modules
interpolation baseline script
evaluation pipeline
training framework outside this task
```

尽量只新增：

```text
src/models/electronic_baseline.py
scripts/train_electronic_baseline.py
```

---

## 5.2 公平性优先于绝对性能

这一步电子 baseline 的核心价值是：

> 在**相似 bottleneck 约束**下，给出一个纯电子参考

因此：

* 不要做“大而强”的 unconstrained network
* 不要把它写成普通 SR SOTA 模型
* 不要为了追分引入大量额外 trick

---

## 5.3 与最终光学主线语义一致

请把这个 baseline 理解为：

```text
electronic encoder + electronic decoder
```

对应论文中的：

```text
electronic encoder + optical decoder
```

也就是说，这个 baseline 不是为了替代 optical model，而是为了回答：

> 如果 decoder 不是光学的，而是纯电子的，在相同 LR bottleneck 下能达到什么程度？

---

# 6. 输出物要求

AI 完成后请同时给出：

## 6.1 新增/修改文件列表

例如：

```text
src/models/electronic_baseline.py
scripts/train_electronic_baseline.py
```

## 6.2 模型结构说明

必须明确说明：

* encoder 如何下采样
* latent shape 是多少
* decoder 如何上采样
* 是否使用 skip connection（默认不允许）

## 6.3 训练方式说明

包括：

* 数据来源
* loss
* optimizer
* 默认超参数
* 输出目录

## 6.4 如何运行

至少给出一条 smoke test 命令，和一条正常训练命令。

## 6.5 输出示例说明

例如：

```text
train loss
val PSNR
val SSIM
saved samples
metrics.json
```

## 6.6 风险与待确认点

例如：

* 当前数据仍是 Stage1/2 dataset shell，不是最终完整论文协议
* latent=1 channel 是为公平性选择，不代表电子网络最优设置
* 当前未实现论文原始 encoder CNN 的完全结构复刻
* 当前 loss 未包含论文 optical efficiency term

## 6.7 完成状态

说明是否满足验收标准。

---

# 7. 不要做的事情

不要：

* 实现 U-Net / EDSR / RCAN 等强大 unconstrained 网络
* 重写一套新的评估系统
* 引入光学传播逻辑
* 提前做 quantization / misalignment / complex-valued 分支
* 引入 GAN / perceptual loss / diffusion / transformer
* 重构整个 dataset pipeline
* 改动其它 Stage1/2 脚本

---

# 8. 验收标准

以下条件全部满足，任务才算完成：

1. 成功新增一个可训练的 pure electronic baseline 模型
2. 模型具有显式 bottleneck，默认 latent 为低分辨率单通道表示
3. 不使用高分辨率 skip connection 绕过 bottleneck
4. 训练脚本可以独立运行
5. 可以输出重建图像、loss、PSNR、SSIM
6. 复用了统一评估入口 `evaluate_batch(...)`
7. 改动范围小且清晰
8. 数据与训练设定有明确说明
9. baseline 语义与最终光学主线保持一致
10. 没有顺手扩写到其它阶段的功能

---

# 9. 推荐默认实现（如果你没有更好但同样受限的方案）

如果没有更合适的轻量结构，请优先实现一个：

```text
BottleneckAutoEncoder96
```

其默认行为：

* input: `[B,1,96,96]`
* latent: `[B,1,24,24]`
* output: `[B,1,96,96]`

这是当前最符合项目阶段目标的实现。

---

# 10. 请按以下格式回复

```text
1. 任务理解
2. 计划新增/修改的文件
3. 模型设计说明
4. 训练脚本说明
5. 如何运行
6. 输出示例说明
7. 风险与待确认点
8. 完成状态（是否满足验收标准）
```

---

## 我额外给你的一个建议

这份 prompt 的核心价值，在于它**把“电子 baseline”定义成了“公平对齐光学主线的 bottleneck baseline”**。这比“随便上一个 SRCNN/UNet”更适合你这个项目。

如果你愿意，我下一条可以继续帮你做两件很有价值的事里的一个：

* 把这份 prompt 再压缩成 **更适合 Codex 的硬约束版**
* 或者先帮你写一份 **我期望的电子 baseline 结构草图与验收 checklist**，方便你后续分析 Codex 反馈
