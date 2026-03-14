# Task1&2-2 AI 编码提示词

（插值 baseline：bilinear / bicubic）

你可以把下面整段直接贴给 AI coding agent。

---

# TASK

Stage1&2-2：实现 interpolation baseline（至少支持 bilinear / bicubic）

---

# 背景

你正在一个“复现 diffractive decoder 超分论文”的 Python 项目中工作。
当前项目仍处于 **Stage 1 / Stage 2 groundwork**，目标不是立刻实现完整光学系统，而是先建立：

* dataset sanity check
* unified evaluation pipeline
* interpolation baseline
* pure electronic baseline

这样做的原因是：在进入光学模型之前，必须先确认数据链路、评估链路和 baseline 都是可信的，否则后续实验结果无法判断是任务难、数据错、评估错，还是光学层本身有问题。

当前项目的基础数据壳是：

```text
HR -> downsample -> LR -> upsample / baseline -> reconstructed HR
```

当前 Stage 1/2 中：

* EMNIST 原图可暂时视为当前阶段的 HR 来源
* LR 是由 HR 人工下采样得到
* upsampled LR / baseline 输出用于检查 baseline 质量

这还是一个 **sanity-check SR shell**，不是最终论文里的 learned optical pattern pipeline。

---

# 任务目标

实现一个**最小可用、低耦合、可直接复用统一评估模块的 interpolation baseline**，用于给出最基础的性能参考线。

至少支持两种模式：

```text
bilinear
bicubic
```

核心用途：

1. 给出最基础 baseline 结果
2. 验证数据链路和评估链路能否联通
3. 为后续 electronic baseline / optical model 提供可比较的下界参考

根据当前项目计划，interpolation baseline 是 Stage 1 的核心子任务之一，也是 Stage 1/2 执行顺序中的第三步。

---

# 你需要完成的具体内容

请实现一个独立脚本，建议路径：

```text
scripts/run_interpolation_baseline.py
```

该脚本应完成：

1. 读取当前阶段的 HR 图像样本
2. 构造 LR 图像（由 HR 下采样得到）
3. 使用指定插值方法将 LR 上采样回 HR 尺寸
4. 调用项目已有统一评估入口，对 baseline 结果计算：

   * PSNR
   * SSIM
5. 保存若干可视化结果
6. 导出本次 baseline 的 summary / metrics 文件

---

# 必须优先复用的已有模块

请优先复用项目中**已经存在**的：

```text
src/eval/metrics.py
src/eval/evaluator.py
```

以及统一入口：

```python
from src.eval.evaluator import evaluate_batch
```

不要重新实现一套平行的 PSNR / SSIM 逻辑。
这一步的 baseline 必须建立在已经完成的统一评估 pipeline 之上，否则会破坏“统一口径”的设计目标。

---

# 实现要求

## 1. 插值 baseline 脚本

新增：

```text
scripts/run_interpolation_baseline.py
```

支持的命令行参数建议至少包括：

```text
--split train|val|test
--num-samples 100
--seed 42
--output-dir outputs/interpolation/train
--mode bilinear|bicubic
--save-grid / --no-save-grid
--save-individual / --no-save-individual
--download
```

如果当前项目尚无统一 dataset pipeline，可以像 dataset inspection 那样，做一个**最小独立实现**，但不要重构整个数据模块，也不要顺手改训练入口。
这符合当前项目“小任务、局部改动、低耦合”的工作流。

---

## 2. 数据处理逻辑

当前阶段按以下壳子执行：

```text
HR -> downsample -> LR -> interpolate -> SR_baseline
```

要求：

* HR 与 LR 必须来自同一样本
* LR 的构造逻辑要与当前 dataset inspection 阶段保持一致
* 上采样目标尺寸必须与 HR 完全一致
* 不要引入神经网络，不要训练，不要优化参数

---

## 3. 插值实现

请使用已有依赖完成，不要引入新库。
优先使用：

```python
torch.nn.functional.interpolate
```

至少支持：

```text
bilinear
bicubic
```

如果输入是单通道灰度图，需要正确处理 shape，避免因为维度错误导致插值失败。

---

## 4. 可视化输出

建议保存以下结果：

### 单样本图

每个样本至少包含：

```text
HR
LR
Interpolated SR
```

如果容易实现，也可以额外显示：

```text
|SR - HR| difference map
sample index
shape
min/max
mode
```

### 总览图

生成一张网格图，例如：

```text
每行一个样本
三列或四列：
HR | LR | Interpolated | diff(optional)
```

注意：

* 灰度图显示要正确
* 避免默认 colormap 误导观察
* 图标题尽量包含样本 index 和 shape

这和你前一个 dataset inspection 脚本的工程风格应保持一致。

---

## 5. 指标导出

必须调用统一评估入口：

```python
metrics = evaluate_batch(preds, targets)
```

并导出至少一个：

```text
metrics.json
```

内容建议包括：

```json
{
  "mode": "bicubic",
  "split": "train",
  "num_samples": 100,
  "seed": 42,
  "psnr_mean": ...,
  "ssim_mean": ...,
  "hr_shape": ...,
  "lr_shape": ...
}
```

如果方便，也可以额外保存：

```text
summary.json
```

记录本次运行参数与样本信息。

---

# 代码结构建议

建议只新增以下文件：

```text
scripts/run_interpolation_baseline.py
```

如果确实需要极小的辅助函数，可以新增很小的工具文件，但不要扩大改动范围。
默认不修改：

```text
models/
train scripts
optical modules
existing eval files
dataset inspection script
```

---

# 设计原则

## 1. 最小改动原则

这是一个 baseline 脚本任务，不是框架重构任务。
不要顺手：

* 改训练代码
* 改模型代码
* 改光学模块
* 重写 dataset pipeline
* 重写 eval pipeline

---

## 2. 测试驱动 / 黑盒验收友好

脚本完成后，应该能通过一条简单命令验证：

```bash
python scripts/run_interpolation_baseline.py --split train --num-samples 32 --seed 42 --output-dir outputs/interpolation/train --mode bicubic
```

运行后应能看到：

* 控制台打印 PSNR / SSIM
* 输出目录自动创建
* 生成 metrics.json
* 生成若干图像

---

## 3. 与当前阶段目标保持一致

这一步不是为了追论文最终指标，也不是为了对齐最终 96×96 大图设定。
它的目标是：

> 验证当前阶段的 baseline、数据和评估链路能跑通，给出最基础参考线。

如果当前仍以 EMNIST 28×28 壳子为主，是合理的；但请把关键设定明确记录进输出 summary 中，避免未来混淆。

---

# 不要做的事情

不要：

* 修改模型文件
* 修改训练脚本
* 添加神经网络
* 重新实现 PSNR / SSIM
* 绑定某个特定网络结构
* 做 optical model 相关逻辑
* 提前实现 electronic baseline
* 提前实现 paper-aligned 96×96 大图拼接逻辑
* 引入复杂配置系统改造

---

# 输出文件要求

AI 完成后，请同时给出：

## 1. 修改/新增了哪些文件

例如：

```text
scripts/run_interpolation_baseline.py
```

## 2. 你对当前数据接口做了什么假设

例如：

* 是否直接读取 EMNIST
* 是否复用 dataset inspection 的数据构造方式
* HR/LR 如何生成
* 当前 split 的定义方式

## 3. 如何运行脚本

给出至少一条可直接复制的命令。

## 4. 输出会生成哪些文件

例如：

```text
outputs/interpolation/train/
  metrics.json
  interpolation_grid.png
  sample_001.png
  sample_002.png
```

## 5. 有哪些潜在风险点或你不确定的地方

例如：

* 当前 HR 定义只是 Stage1/2 sanity-check 版本
* LR 构造尺度是否完全与后续 baseline / train 保持一致
* bicubic 是否会出现轻微越界
* batch 输入 shape 是否需要显式扩维

## 6. 完成状态（是否满足验收标准）

---

# 验收标准

只有以下条件全部满足，任务才算完成：

1. 脚本可以独立运行
2. 至少支持 bilinear 与 bicubic 两种模式
3. 能从 HR 构造 LR，并上采样回 HR 尺寸
4. 能正确调用统一评估接口 `evaluate_batch(...)`
5. 能输出 PSNR / SSIM 指标
6. 能保存总览图或单样本图中的至少一种
7. 输出目录自动创建
8. 不修改训练/模型/光学主流程
9. 改动范围小且清晰
10. 对当前阶段的数据设定与潜在限制有明确说明

---

# 建议脚本接口

尽量支持类似：

```bash
python scripts/run_interpolation_baseline.py \
  --split train \
  --num-samples 64 \
  --seed 42 \
  --output-dir outputs/interpolation/train \
  --mode bicubic
```

也建议支持：

```bash
python scripts/run_interpolation_baseline.py \
  --split val \
  --num-samples 64 \
  --seed 42 \
  --output-dir outputs/interpolation/val \
  --mode bilinear \
  --save-grid \
  --no-save-individual
```

---

# 额外说明

这个任务的重点不是“写出很炫的代码”，而是：

* 给当前项目建立可信的 baseline 下界
* 把统一评估模块真正接入一次真实任务
* 为后续 pure electronic baseline 和 optical model 提供统一比较参考

按照项目当前路线，interpolation baseline 的作用就是“确认评价流程正常，并给出最基础参考线”；如果后续 pure electronic baseline 明显优于它，就说明任务与训练链路至少是有希望的。

---

# 请按以下格式回复

```text
1. 任务理解
2. 计划修改的文件
3. 实现说明
4. 如何运行
5. 输出示例说明
6. 风险与待确认点
7. 完成状态（是否满足验收标准）
```
