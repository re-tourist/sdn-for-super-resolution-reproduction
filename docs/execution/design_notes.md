# Design Notes

记录当前实现中的重要设计决策与限制。
这些内容不是 bug，但会影响未来模块开发。

---

## Evaluation Pipeline

### 1. Tensor shape ambiguity

当前 evaluation pipeline 支持以下输入：

- [H,W]
- [C,H,W]
- [N,C,H,W]

对于三维输入：


[N,H,W] / [C,H,W]


由于维度本身不可区分，当前实现采用启发式规则：


如果首维 ∈ {1,2,3,4}
→ 视为 channel 维


因此：


(2,28,28)


可能被解释为 2-channel image 而不是 batch=2。

建议未来调用 evaluation API 时统一使用：


[N,C,H,W]


以避免歧义。

### 2. Input range handling

evaluation pipeline 假设输入图像范围为：


[0,1]


当前实现的容错策略：

| 输入范围 | 行为 |
|---|---|
| [0,1] | 直接使用 |
| [0,255] | 自动除以 255 |
| 其它 | clip 到 [0,1] |

这种策略方便工具脚本，但可能掩盖上游错误。

未来正式实验建议：

- 明确保证输入范围为 `[0,1]`
- 或在 evaluation 中改为严格报错

## Baseline

### 1. Interpolation Baseline Design Notes

当前 Stage1/2 中实现的 interpolation baseline 主要用于：

- 验证数据链路是否正确
- 验证统一评估 pipeline 是否可用
- 提供最基础的性能参考下界

该 baseline 属于 **sanity-check baseline**，并不代表最终论文对齐的实验协议。因此在当前实现中存在一些需要记录的设计约定与潜在限制。

---

#### 2.1 HR / LR 构造协议（Stage1/2 版本）

当前 baseline 默认使用：

```

hr_size = 96
scale = 4

```

构造流程为：

```

HR -> bicubic downsample -> LR -> interpolation -> SR

```

具体步骤：

1. 原始 EMNIST 图像首先被 resize 到 `hr_size`
2. 使用 **bicubic 下采样**生成 LR 图像
3. 再通过指定插值方法（bilinear / bicubic）上采样回 HR 尺寸

需要注意：

- 当前 HR 定义是 **Stage1/2 的工程壳设定**
- 并不代表论文最终实验所使用的 HR/LR 数据构造协议

未来在进入 **paper-aligned experiments** 时，HR 定义可能会发生调整，因此当前 baseline 指标不应直接与论文结果进行对比。

---

#### 2.2 LR 退化模型假设

当前实现中，LR 图像由 **bicubic 下采样**生成：

```

LR = BicubicDownsample(HR)

```

这意味着当前 baseline 默认假设：

```

degradation model = bicubic

```

因此：

- bilinear baseline 与 bicubic baseline 实际是在同一退化模型下进行比较
- 不代表任意 LR 来源的通用性能下界

这一设定符合常见 SR baseline 做法，但应在实验记录中明确说明。

未来如果需要测试：

- 不同退化模型
- 更接近真实光学系统的 degradation

则需要重新定义 LR 构造方式。

---

#### 2.3 内部验证集（val split）

当前实现中的 `val` split 并非官方数据集划分。

如果 EMNIST 未提供明确的验证集，则脚本会：

```

从 train split 中按固定 seed 划分出 val

```

其目的仅为：

- 在 Stage1/2 阶段进行 baseline 验证
- 确认评估 pipeline 工作正常

该 val split 不应被视为最终实验协议中的验证集。

---

#### 2.4 插值导致的数值越界

在某些情况下：

```

bicubic / bilinear interpolation

```

可能产生轻微数值越界，例如：

```

pixel < 0
pixel > 1

```

当前 baseline 依赖统一评估 pipeline 对输入进行范围处理，使最终指标计算保持在 `[0,1]` 口径。

该策略在工具脚本阶段是可接受的，但需要注意：

- 评估模块当前包含一定的容错策略
- 后续正式实验阶段应尽量保证输入范围严格符合 `[0,1]`

---

#### 2.5 Baseline 的阶段性性质

Interpolation baseline 的设计目标是：

```

提供任务的最基础性能参考下界

```

如果后续 **pure electronic baseline** 的性能没有明显优于 interpolation baseline，则需要重新检查：

- 数据 pipeline
- LR / HR 构造逻辑
- 评估指标实现

因此 interpolation baseline 的结果在当前项目中主要用于：

```

pipeline sanity check

```

而不是用于最终论文级性能对比。

## Dataset Construction

## Optical Propagation Implementation

## Experiment Protocol 
