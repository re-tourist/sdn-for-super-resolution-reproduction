# Stage 1 & Stage 2 工程执行计划

## 一、阶段总体目标

### Stage 1 目标

建立**可信的 baseline 与最小训练闭环验证能力**。

核心问题：

* 数据是否正确
* 任务是否可学
* 评价指标是否可靠
* baseline 表现如何

如果这些没有确认，后续光学网络的实验结果**没有可信度**。

---

### Stage 2 目标

稳定并标准化：

* 数据 pipeline
* 评估 pipeline
* 可视化 pipeline
* 实验记录方式

最终达到：

> 任意实验运行 → 自动得到可比较结果。

---

# 二、阶段整体执行顺序

**推荐执行顺序**

```
Task1&2-1   数据集检查
Task1&2-3   统一评估 pipeline
Task1&2-2   插值 baseline
Task1&2-4   纯电子 baseline
Task1&2-5   单样本过拟合实验
Task1&2-6   小数据集训练
Task1&2-7   文档与实验记录
```

该顺序遵循一个原则：

> 先确认数据 → 再确认指标 → 再跑 baseline → 再验证模型可学。

---

# 三、Task 1&2-1

# 数据集准备与数据抽样可视化检查

对应 Issue
`[Task1&2-1] 数据集准备和数据抽样可视化检查`

### 目标

确认数据 pipeline 没有问题：

* HR/LR 构造正确
* downsample/upsample 正确
* 图像 orientation 正确
* patch 裁剪正确

---

### 输入

原始数据：

```
EMNIST
```

当前 pipeline：

```
HR → downsample → LR → upsample → reconstruction
```

---

### 需要完成的功能

实现脚本：

```
scripts/dataset_sanity_check.py
```

功能：

1. 随机抽样 HR 图像
2. 自动生成 LR
3. 生成插值图像
4. 保存对比图

---

### 输出示例

```
outputs/sanity_check/

sample_01_hr.png
sample_01_lr.png
sample_01_bicubic.png

grid_sample_01.png
```

grid 图结构：

```
HR | LR | Bicubic
```

---

### 验证标准

必须人工检查：

* HR 是否清晰
* LR 是否正确模糊
* bicubic 是否合理

---

### 完成标志

满足以下条件：

* 随机样本显示正确
* HR/LR 对齐
* 尺寸正确
* 没有旋转错误

---

# 四、Task 1&2-3

# 统一 PSNR / SSIM 评估 Pipeline

对应 Issue
`[Task1&2-3] 创建统一的PSNR/SSIM评估 pipeline`

---

### 目标

实现统一的评估模块：

```
src/eval/metrics.py
```

避免：

* 每个脚本各自计算指标
* PSNR 实现不一致
* SSIM 参数不同

---

### 需要实现

统一函数：

```
compute_psnr(pred, target)
compute_ssim(pred, target)
```

要求：

* 支持 batch
* 支持 numpy / torch

---

### 推荐实现

PSNR

```
PSNR = 10 log10( MAX^2 / MSE )
```

SSIM

使用：

```
skimage.metrics.structural_similarity
```

---

### 统一评估入口

实现：

```
src/eval/evaluator.py
```

功能：

```
evaluate_batch(preds, targets)
```

返回：

```
{
  psnr_mean
  ssim_mean
}
```

---

### 输出记录

保存：

```
outputs/eval/

metrics.json
```

---

# 五、Task 1&2-2

# 插值 Baseline

对应 Issue
`[Task1&2-2] 实现双线性与双三次插值 baselines`

---

### 目标

建立**最基础性能下界**。

如果模型比 bicubic 差：

说明 pipeline 或训练有问题。

---

### 需要实现

插值方法：

```
bilinear
bicubic
```

使用：

```
torch.nn.functional.interpolate
```

---

### 脚本

```
scripts/run_interpolation_baseline.py
```

---

### 输入

```
LR image
```

---

### 输出

```
SR image
```

---

### 保存结果

```
outputs/interpolation/

sample_grid.png
metrics.json
```

---

### Grid 示例

```
HR | LR | Bilinear | Bicubic
```

---

### 记录指标

```
PSNR
SSIM
```

---

### 完成标志

得到 baseline 指标：

```
PSNR_bilinear
PSNR_bicubic
```

这些将成为**后续实验参考线**。

---

# 六、Task 1&2-4

# 纯电子 Baseline 最小模型

对应 Issue
`[Task1&2-4] 纯电子 baseline 最小模型`

---

### 目标

验证：

> 任务本身是否可学。

---

### 推荐模型

简单 CNN：

```
SRCNN style
```

结构：

```
LR
 ↓
Conv 5x5
 ↓
ReLU
 ↓
Conv 3x3
 ↓
ReLU
 ↓
Conv 3x3
 ↓
SR
```

---

### 实现文件

```
src/models/electronic_baseline.py
```

---

### 训练脚本

```
scripts/train_electronic_baseline.py
```

---

### 训练配置

```
epoch = 10
batch = 32
lr = 1e-3
loss = L1
```

---

### 输出

```
outputs/electronic_baseline/

checkpoints/
metrics.json
samples/
```

---

### 完成标志

确认：

```
PSNR_electronic > PSNR_bicubic
```

否则说明：

* pipeline 有问题
* 或训练不稳定

---

# 七、Task 1&2-5

# 单样本过拟合实验

对应 Issue
`[Task1&2-5] 单样本过拟合实验`

---

### 目标

验证：

> 模型是否能学习。

---

### 方法

训练：

```
只用 1 张图
```

训练：

```
500~1000 steps
```

---

### 期望现象

loss：

```
持续下降
```

输出：

```
越来越接近 HR
```

---

### 如果失败

说明：

* 模型结构问题
* loss 不对
* 梯度问题

---

# 八、Task 1&2-6

# 小数据集训练闭环

对应 Issue
`[Task1&2-6] 小数据集训练闭环`

---

### 目标

确认完整训练 pipeline 可运行。

---

### 数据规模

```
100 ~ 500 samples
```

---

### 训练设置

```
epoch 20
batch 16
```

---

### 需要验证

训练过程中：

```
loss 是否下降
PSNR 是否上升
```

---

### 输出

```
training_curve.png
```

---

# 九、Task 1&2-7

# 文档与实验记录更新

对应 Issue
`[Task1&2-7] 阶段1、2文档与实验记录更新`

---

### 更新文档

```
docs/execution/experiment_log.md
```

记录：

```
baseline results
training config
dataset info
```

---

### 更新结果汇总

```
docs/execution/results_summary.md
```

记录：

```
baseline PSNR
baseline SSIM
```

---

### 更新 checklist

```
docs/execution/checklist.md
```

---

# 十、Stage 1/2 完成标准

完成以下条件即可进入 Stage3：

---

### 数据可信

```
dataset_sanity_check ✔
```

---

### 指标可信

```
psnr / ssim pipeline ✔
```

---

### baseline 正常

```
bicubic baseline ✔
electronic baseline ✔
```

---

### 模型可学

```
single sample overfit ✔
```

---

### 训练 pipeline 正常

```
small subset training ✔
```

---

# 十一、Stage 1 / 2 关键输出

最终你应该拥有：

```
baseline_metrics.json
training_curves.png
sanity_check_images/
baseline_samples/
```

以及文档：

```
experiment_log.md
results_summary.md
```

---

# 十二、进入 Stage3 的条件

当 Stage1/2 完成后，可以进入：

```
Stage3 optical module
```

包括：

```
diffraction propagation
phase mask
optical decoder
```

但必须满足：

> baseline pipeline 已完全可信。

