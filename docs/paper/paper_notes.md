# Paper Notes — Super-resolution image display using diffractive decoders

## 1. 论文核心一句话

这篇论文提出了一个 **电子编码器 + 全光衍射解码器** 的联合训练框架：
前端 CNN 将高分辨率目标图像编码成低分辨率 SLM 调制 pattern，后端 diffractive decoder 从该 low-resolution pattern 直接在输出 FOV 合成 super-resolved 图像。

可以把它理解为：

- 前端：digital encoder
- 后端：all-optical decoder
- 系统整体：hybrid autoencoder / learned computational display

---

## 2. 复现目标范围（当前拍板版）

### 先复现
- 数值实验主线
- phase-only 输入调制
- 电子编码器 + 衍射解码器联合训练
- optical decoder 的层数对比（L=1/3/5；先保证 paper-aligned configs 可运行，系统化比较后置到 Stage 6）
- PSNR / SSIM 指标
- blind 高频 line-pair test（在 paper-aligned pipeline 就绪后进入 Stage 5）
- 量化 sweep（至少推理阶段测试；属于总复现范围，但按当前阶段计划后置到 Stage 6）

### 暂不优先
- 真实 THz 实验系统
- 3D 打印层厚映射到 STL
- detector 扫描链路
- full hardware-in-the-loop
- 多波长 / RGB 扩展

---

## 3. 论文结构理解

### 3.1 电子编码器做什么
- 输入：高分辨率目标图像
- 输出：低分辨率 modulation pattern
- 该 pattern 不是普通低清图，而是给 SLM / wavefront modulator 使用的 **光学友好低维表示**
- 主线版本中，这个 pattern 是 **phase-only LR representation**

### 3.2 光学解码器做什么
- 输入：由 LR pattern 生成的 coherent wave
- 网络：1 / 3 / 5 层 transmissive diffractive layers
- 输出：输出 FOV 上的强度图像
- 本质：从低维 pattern 到目标图像的全光解码器

### 3.3 为什么这篇论文重要
它不是简单做“光学超分”，而是：

- 不依赖传统迭代 hologram optimization
- 用 jointly-trained encoder-decoder 学习更适合光学传播的 pattern
- 让低 SBP 的 wavefront modulator 结合 diffractive decoder 实现更高的有效 SBP

---

## 4. 数据集与任务设定

### 4.1 数值实验数据集
- 来自 EMNIST letters
- 原始字符大小：28×28
- 先 bicubic 插值到 32×32
- 再通过 tiling 组成 96×96 图像

### 4.2 数值实验数据量
- train: 60,000 张 96×96 图像
- val: 6,000 张 96×96 图像
- test: 6,000 张 96×96 图像

### 4.3 图像内容构造
- train/val 图像中包含 1,2,3,4 个 handwritten letters
- test 图像中包含 6,7,8,9 个 handwritten letters

### 4.4 blind test
- 使用 line pairs / resolution targets 进行盲测
- 这些 line targets 不在训练集中
- 作用：验证是否真的恢复了高频结构，而不是只记住 letters 统计模式

---

## 5. 光学前向模型

### 5.1 衍射层表示
每个衍射层离散为规则 2D 网格，网格点称为 diffractive neuron。

层的复透射率：

\[
 t_l[m,n] = \exp\left(j\frac{2\pi}{\lambda}(\tau(\lambda)-n_a)h_l[m,n]\right)
\]

在数值实验中，假设吸收为 0，可简化为纯 phase modulation：

\[
 t_l[m,n] = e^{j\theta_l[m,n]}
\]

### 5.2 自由空间传播
- 使用 FFT-based angular spectrum / Rayleigh-Sommerfeld diffraction integral
- 相邻层之间传播写成 convolution 或频域乘法

### 5.3 你实现时的核心链路
- encoder 输出 `phi_lr`
- 构造输入复场 `U0 = exp(j * phi_lr)`
- 每层：`U <- U * T_l`，再做 `propagate(U)`
- 输出强度：`I = |U_out|^2`

---

## 6. 数值实验关键设置

### 6.1 采样与尺寸
- diffractive neuron width / sampling period: 0.533 λ
- 每层大小：200×200 pixels，对应 106.66λ × 106.66λ
- 输入/输出 FOV：96×96 pixels，对应 51.168λ × 51.168λ
- 为避免 aliasing，zero padding 到 400×400

### 6.2 diffractive decoder 层数
- L = 1, 3, 5

### 6.3 axial distances（数值实验）
- 1 layer: d1 = 6.667λ, d3 = 173.333λ
- 3 layers: d1 = 4λ, d2 = 53.334λ, d3 = 53.334λ
- 5 layers: d1 = 2.667λ, d2 = 66.667λ, d3 = 80λ

> 注：实现时需明确这些距离在代码中的确切对应关系（输入面到第1层、第1层到第2层……、最后一层到输出面）。

### 6.4 初始化
- 数值实验中，各层 phase coefficients 初始化为 0

---

## 7. 输入调制方式

### 7.1 主线：phase-only
- 论文主图和主要数值结果都强调 phase-only SLM
- encoder 生成 phase-only LR representations

### 7.2 对照：complex-valued
- complex-valued 输入平均略优于 phase-only
- 原因：自由度更高
- 但 phase-only 已经是主线和更贴近器件的版本

### 7.3 还有 amplitude-only 补充结果
- 论文补充材料提到 amplitude-only encoder 结果
- 当前阶段不优先复现

---

## 8. Loss 与训练策略

### 8.1 训练损失
联合训练时使用：

\[
\mathcal L = \frac{1}{N}\sum_i |y_i - \sigma \hat y_i| + \gamma e^{-\eta}
\]

其中：

\[
\sigma = \frac{\sum_i y_i}{\sum_i \hat y_i}
\]

\[
\eta = 100 \times \frac{P_o}{P_i}
\]

含义：
- 第一项：MAE（带归一化）
- 第二项：效率项，鼓励输出 FOV 内光能更高

### 8.2 efficiency penalty 使用范围
- 实验演示的 decoder 开启该项：
  - γ = 0.005 for L=1
  - γ = 0.015 for L=3
- 其他设计 γ = 0

这意味着：
- efficiency term 很重要
- 但并非所有数值实验都必须开
- 复现时可以把它做成可开关 ablation

### 8.3 数据增强
训练中使用：
- random rotations: 0, 90, 180, 270 degrees
- random flipping
- random contrast adjustments

### 8.4 优化器与训练配置
- Python 3.6.12
- TensorFlow 1.15.4
- Adam
- lr(decoder) = 0.001
- lr(encoder) = 0.0005
- batch size = 40
- 500 epochs

> 你复现时可用 PyTorch，但要尽量保持这些训练超参量级一致。

---

## 9. 评估指标与 baseline

### 9.1 指标
- PSNR
- SSIM

### 9.2 图中 LR baseline
论文图里的 LR 对照图像是：
- 采用 bicubic kernel
- 带 anti-aliasing
- 做 k-fold downsampling 得到

所以你的 baseline 不能随便做 nearest downsample。

---

## 10. 主要实验结论

### 10.1 deeper decoder 更强
- L=5 > L=3 > L=1（整体趋势）
- 单层 decoder 的 generalization 更有限

### 10.2 complex-valued 略优于 phase-only
- 但 phase-only 仍可行且是主线

### 10.3 blind 高频 line test 很关键
- 5-layer phase-only decoder 能恢复更细的 line pairs
- 1-layer decoder 在更难条件下会失败

### 10.4 SBP 提升未达到理论 k^2
论文自己解释：
- 训练数据（handwritten letters）高频结构代表性不足
- line / grating 不在训练集中

这对复现的启发：
- 若 blind 高频测试不佳，不一定是代码错，也可能是训练数据频谱覆盖不足

---

## 11. 量化分析

### 11.1 论文做法
- 用 16-bit phase quantization 训练 phase-only modulator
- 测试时 blind test 到 8, 6, 4, 2 bit

### 11.2 结论
- 8-bit 和 6-bit 几乎不掉
- 4-bit 明显退化
- 2-bit 无法清晰合成输出
- 4/2 bit 情况下，可以从头训练来改善性能

### 11.3 复现建议
- 第一版先不做 quantization-aware training
- 先做 continuous / high-bit 训练
- 然后做 16/8/6/4/2 bit 推理测试

---

## 12. 实验版 THz 系统（先了解，后复现）

### 12.1 波长
- λ ≈ 0.75 mm

### 12.2 尺寸
- layer size: 5 cm × 5 cm
- output FOV size: 3 cm × 3 cm
- output image size: 15×15
- LR modulator: 5×5 phase-only layer
- SR factor: k = 15/5 = 3

### 12.3 实验版 zero padding
- 300×300

### 12.4 材料参数
- measured complex refractive index: ~1.6518 + j0.0612

### 12.5 misalignment vaccination
训练时加入随机 3D displacement：
- Δx ≈ 0.334 λ
- Δy ≈ 0.334 λ
- Δz ≈ 0.533 λ

> 当前阶段不优先复现真实实验链，但这些参数要记下来，后面做 robustness test 时可参考。

---

## 13. 当前实现优先顺序（拍板版）

### 第一步
- 数值实验
- phase-only
- L=1/3/5
- EMNIST 96×96 数据构造
- zero padding
- MAE 主损失
- baseline 先做 bicubic + 纯电子模型

### 第二步
- blind line-pair test（Stage 5，在 paper-aligned pipeline 建立后执行）

### 第三步
- efficiency on/off ablation（Stage 6）
- complex-valued 对照（Stage 6）
- quantization sweep（Stage 6）

### 第四步（以后）
- misalignment test / vaccination
- THz experimental setting closer reproduction

---

## 14. 当前仍未完全确定的实现细节

这些都要在复现过程中逐步拍板并写入 experiment_log：

1. encoder 的具体层结构是否严格按文中描述实现，还是先用简化版 stride-conv 实现
2. phase-only pattern 的范围约束是否在训练时显式映射到 [0, 2π)
3. σ 归一化项是严格按论文写进 loss，还是用等价实现
4. output FOV crop 在代码里怎么和 full propagation grid 对齐
5. 不同 axial distances 在 forward 里的顺序和对应关系
6. blind line target 评估图如何生成
7. 是否需要先做单样本 phase-pattern optimization 测试，再接 encoder

---

## 15. 我们当前的执行策略

一句话版本：

**先用数值实验主线，把“96×96 图像 -> 低分辨率 phase pattern -> diffractive decoder -> 输出 FOV 强度图”这条链完整跑通，再逐步补 blind 高频测试、量化分析、错位鲁棒性和复杂调制对照。**
