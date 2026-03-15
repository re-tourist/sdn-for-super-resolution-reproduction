# Task1&2-3 AI 编码提示词

（统一 PSNR / SSIM 评估 pipeline）

# TASK

Stage1&2-3：实现统一 PSNR / SSIM 评估 pipeline

---

# 背景

当前项目正在复现论文：

> Super-resolution image display using a diffractive optical network

当前开发阶段是：

**Stage 1 / Stage 2 groundwork**

目标是先建立：

* baseline
* evaluation pipeline
* data sanity checks

然后才进入光学模块实现。

目前已经完成：

```
Task1&2-1
dataset inspection script
```

接下来需要实现：

```
Task1&2-3
统一 PSNR / SSIM 评估 pipeline
```

原因：

后续多个模块都需要统一指标：

* interpolation baseline
* electronic baseline
* optical model
* ablation experiments

如果每个脚本各自计算 PSNR / SSIM，会导致：

* 口径不一致
* 实验结果不可比较
* debug 困难

因此需要一个**统一的评估模块**。

---

# 任务目标

实现一个**统一、可复用、低耦合的 evaluation pipeline**。

核心功能：

```
输入:
    SR prediction
    HR ground truth

输出:
    PSNR
    SSIM
```

支持：

* 单张图
* batch
* numpy / torch tensor

并提供：

```
evaluate_batch(...)
```

作为统一入口。

---

# 功能要求

实现以下模块：

```
src/eval/metrics.py
src/eval/evaluator.py
```

---

# 1 metrics.py

实现：

```
compute_psnr(pred, target)
compute_ssim(pred, target)
```

要求：

### 输入格式

支持：

```
[H,W]
[1,H,W]
[C,H,W]
[N,C,H,W]
```

支持：

```
torch.Tensor
numpy.ndarray
```

内部需要统一为：

```
float32
range = [0,1]
```

---

### PSNR 公式

使用标准公式：

```
PSNR = 10 * log10( MAX^2 / MSE )
```

其中：

```
MAX = 1
```

因为项目统一使用：

```
image range = [0,1]
```

---

### SSIM

使用：

```
skimage.metrics.structural_similarity
```

参数建议：

```
data_range=1.0
```

---

### 注意事项

需要正确处理：

* 灰度图
* batch
* channel 维

SSIM 计算时建议：

```
逐图计算
然后取平均
```

---

# 2 evaluator.py

实现统一接口：

```
evaluate_batch(preds, targets)
```

输入：

```
preds   : SR images
targets : HR images
```

支持：

```
numpy
torch
batch
```

返回：

```
{
    "psnr_mean": float,
    "ssim_mean": float,
    "num_samples": int
}
```

---

# 3 CLI测试脚本

新增脚本：

```
scripts/test_eval_pipeline.py
```

功能：

* 随机生成 HR 图
* 添加噪声生成 SR
* 运行 evaluation pipeline
* 打印指标

例如：

```
PSNR: 22.1
SSIM: 0.65
```

用于 sanity check。

---

# 代码结构

建议新增目录：

```
src/
  eval/
    metrics.py
    evaluator.py
```

脚本：

```
scripts/test_eval_pipeline.py
```

---

# 设计原则

必须遵守：

### 1 最小改动原则

不要修改：

```
models/
train scripts
dataset scripts
```

只新增：

```
src/eval/
scripts/test_eval_pipeline.py
```

---

### 2 可复用

后续模块都要调用：

```
evaluate_batch(...)
```

例如：

```
interpolation baseline
electronic baseline
optical model
```

---

### 3 无训练依赖

evaluation pipeline 不应依赖：

```
training framework
optimizer
model
```

应是纯函数式工具。

---

# 输出文件

AI 完成后应给出：

### 1 新增文件

```
src/eval/metrics.py
src/eval/evaluator.py
scripts/test_eval_pipeline.py
```

---

### 2 示例运行方法

例如：

```
python scripts/test_eval_pipeline.py
```

---

### 3 示例输出

例如：

```
PSNR: 20.31
SSIM: 0.71
samples: 8
```

---

# 验收标准

以下条件全部满足才算通过。

---

### 1 模块可导入

```
from src.eval.evaluator import evaluate_batch
```

能够正常运行。

---

### 2 支持 batch

例如：

```
N,C,H,W
```

输入不会报错。

---

### 3 支持 numpy / torch

以下两种都能运行：

```
torch tensor
numpy array
```

---

### 4 CLI 测试脚本可运行

```
python scripts/test_eval_pipeline.py
```

能够输出：

```
PSNR
SSIM
```

---

### 5 不修改现有代码

只新增：

```
src/eval
scripts/test_eval_pipeline.py
```

---

# 不要做的事情

AI 不要：

* 修改训练代码
* 修改 dataset inspection 脚本
* 引入复杂依赖
* 实现训练逻辑
* 绑定特定模型

这是一个**通用 evaluation 工具模块**。

---

# 风险提示

注意以下潜在问题：

### 1 Tensor 维度

```
[C,H,W]
[N,C,H,W]
```

需要统一处理。

---

### 2 range

确保：

```
image range = [0,1]
```

否则 PSNR 会错误。

---

### 3 SSIM

SSIM 对输入维度较敏感：

需要确保输入：

```
[H,W]
```

或

```
[H,W,C]
```

---

# 最终目标

完成后，你应该能在任意地方调用：

```
from src.eval.evaluator import evaluate_batch

metrics = evaluate_batch(preds, targets)
```

并得到：

```
PSNR
SSIM
```

作为**统一评估接口**。

---

# 任务结束输出

AI 完成任务时需要提供：

1️⃣ 修改文件列表
2️⃣ 关键代码片段
3️⃣ CLI 运行示例
4️⃣ 可能的风险说明


