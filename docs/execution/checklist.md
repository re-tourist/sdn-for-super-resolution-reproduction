# 一、训练前 checklist

## 1. 先确认你复现的是哪一条主线

这篇论文的主线是：

* **电子编码器输出低分辨率 SLM modulation pattern**
* **光学部分是 all-optical diffractive decoder**
* **主要 phase-only，complex-valued 是对照/扩展**
* **数值实验与实验系统的设置不完全一样**

所以你一开始要先拍板：

* 先复现**数值实验**
* 还是直接复现实验版 THz 系统

建议先做**数值实验 phase-only 主线**。

---

## 2. 搞清楚“低分辨率 pattern”不是普通低清图

论文里的 encoder 输出不是给人看的小图，而是**低分辨率 SLM modulation pattern**，本质是给波前调制器/SLM 用的低维表示。

训练前你要确认：

* 你的 encoder 输出到底是 `phase map`
* 还是 `amplitude + phase`
* 不要误把它当普通 downsample 图像

---

## 3. 数据集尺寸、任务尺寸要先对齐

论文数值实验用的是拼接后的 `96×96` EMNIST 图像；实验版则是 `15×15` 输出图像，对应 `5×5` 的 phase-only LR modulator。

训练前必须明确：

* 输入图尺寸
* 输出图尺寸
* LR pattern 尺寸
* 目标 SR factor (k)

不然你后面很容易把 encoder、传播网格、输出 ROI 搞混。

---

## 4. 传播网格和输出 FOV 要分开想

论文数值实验里：

* 每层衍射面是 `200×200`
* 输入/输出 FOV 是 `96×96`
* 为了避免 aliasing，矩阵又 zero padding 到 `400×400`

这说明你训练前要明确三区分：

* **pattern 尺寸**
* **真正成像/监督的 FOV**
* **内部传播计算网格**

不要把这三者当成同一个尺寸。

---

## 5. 零填充不是可有可无

论文明确说为了避免 spatial aliasing，做了 zero padding：数值实验 `400×400`，实验版 `300×300`。

训练前必须检查：

* 你有没有 padding
* padding 后的频域坐标是不是还和 `dx, dy` 匹配
* 你监督的是 crop 后 FOV，不是 padding 后整张图

这一项很容易被忽略，但非常影响传播仿真是否可信。

---

## 6. phase-only 版本先跑通

论文主线大量结果都基于 phase-only；complex-valued 只是说明自由度更高、平均略好。

所以训练前建议：

* 第一版只做 `U0 = exp(jφ)`
* 不要一开始就加 amplitude 分支

这样更容易排错。

---

## 7. 初始化方式先定清楚

论文数值实验里，decoder 各层 phase coefficient 初始化为 0。

你复现时要先明确：

* 是严格照论文全 0 初始化
* 还是用小随机初始化做工程稳健版

如果你是“论文忠实复现”，先照论文来。
如果你是“快速稳定跑通”，小随机初始化也值得试，但要单独记录。

---

# 二、训练中 checklist

## 1. loss 不是只有图像误差

论文联合训练用的是：

* MAE
* 加一个 efficiency penalty (e^{-\eta})

其中 (\eta) 由输出 FOV 功率和输入 FOV 功率定义。

训练中必须检查：

* 你是否只用了 L1/MAE
* 是否需要加 efficiency term
* 你的 `η` 统计的是不是**输出 FOV 内**的功率，而不是整张传播平面

---

## 2. efficiency penalty 不是“装饰项”

论文里对实验版 decoder，专门打开了这个效率正则；而对其他设计，(\gamma=0)。

这告诉你两点：

* 它**不是所有情况都必须开**
* 但在更偏真实系统、想要更好输出能量利用时，非常有用

训练中建议你：

* 先做一版 `γ=0`
* 再做一版 `γ>0`
* 对比 ROI 亮度、背景散能、PSNR/SSIM

---

## 3. 训练时要做数据增强

论文明确用了：

* 随机旋转 (0,90,180,270^\circ)
* 随机翻转
* 随机对比度调整

训练中要检查：

* 你是不是完全没做增强
* 如果结果过拟合或泛化差，优先补这些增强，而不是先瞎改网络

---

## 4. 分组学习率

论文用 Adam，且：

* decoder 学习率 `0.001`
* encoder 学习率 `0.0005`

训练中建议你检查：

* encoder 和 decoder 有没有分组学习率
* 不要默认整个模型一个 lr

这在半光模型里比普通 CNN 更重要。

---

## 5. 训练时先做单样本过拟合

这不是论文原文要求，但对复现极重要。

训练中你要先确认：

* 1 张图能不能被系统明显记住
* loss 是否稳定下降
* optical phase 参数有没有非零梯度

如果单样本都过拟合不了，先别上完整数据集。

---

## 6. 训练时要看整张输出平面，不只看中心 crop

因为你真正监督的是 output FOV，但传播是在更大平面上发生的。
如果只看 crop，你可能看不见：

* 能量跑边缘
* 背景强散斑
* 旁瓣过大

训练中建议固定可视化：

* `I_full`
* `I_crop`
* `encoded phase pattern`

---

## 7. 注意 normalization 项

论文 loss 里用了一个 (\sigma) 归一化项，按目标图和预测图的总和来校正。

训练中要检查：

* 你是否完全没做幅值归一化
* 还是你用了和论文不同的 per-image max normalization
* 两者不要混着用却不记录

---

# 三、结果分析 checklist

## 1. 不要只报最终图，要分开看三类图

至少同时看：

* 目标图
* encoder 输出的 LR phase pattern
* optical decoder 输出强度图

论文图里就是这么展示的：低分辨率 phase-only encoded representations、decoder 输出、以及对应 LR 对照。

如果你只看最终结果，很难判断问题出在：

* encoder
* optics
* 还是读出/归一化

---

## 2. 一定要做层数消融

论文反复比较了 `L=1,3,5`，且总体结论很明确：更深的 decoder 更好。

结果分析时至少要比较：

* 1 层
* 3 层
* 5 层

否则你没法判断你自己的系统是：

* 容量不够
* 还是训练没调好

---

## 3. phase-only vs complex-valued 要做成对照，不要混着讲

论文给出的结论是：complex-valued 平均略优，因为自由度更高。

结果分析时要注意：

* 先把 phase-only 主线做稳
* 再拿 complex-valued 当 upper bound / ablation
* 不要一开始两者混做导致排错困难

---

## 4. 别只看自然测试图，还要看线条/高频 blind test

论文专门做了 blind line target 测试，指出训练集高频不足会限制最终可恢复分辨率。

结果分析时建议你额外做：

* 线条
* 条纹
* 简单几何边缘

因为这类图最容易暴露：

* 高频恢复不够
* decoder 容量不足
* 训练集频谱覆盖不够

---

## 5. 不要把“图像像了”直接等同于“物理上好”

论文实验版明确提到，还会看 output diffraction efficiency，且给了平均效率数值。

所以结果分析至少还要看：

* ROI 内能量占比
* 背景能量
* 输出对比度

而不是只盯 PSNR/SSIM。

---

## 6. 训练数据频谱覆盖要单独反思

论文自己就承认：SBP 提升没达到 (k^2) 的一个原因，是训练图像集对更高分辨率特征代表性不够。

所以如果你复现后发现：

* 字母还行
* 线条高频不行

先别急着怪代码，先检查：

* 训练数据是不是也缺高频结构
* 你的增强有没有补足这类模式

---

# 四、硬件鲁棒性 checklist

## 1. 量化位数一定要测

论文做了 phase quantization analysis：

* 16-bit 训练
* blind test 到 8/6/4/2-bit
* 8-bit 和 6-bit 几乎不掉
* 4-bit 明显变差
* 2-bit 基本失败

所以硬件鲁棒性部分必须检查：

* 你有没有做 bit-depth sweep
* 至少测 `16, 8, 6, 4, 2 bit`

这对“复现高质量”非常关键。

---

## 2. misalignment vaccination 不是可有可无

论文实验版明确说：

* 对 layer-to-layer misalignment 做了 vaccination
* 训练时随机引入 x、y、z 三维错位
* 实验版给了扰动量级：(\Delta x,\Delta y \approx 0.334\lambda)，(\Delta z \approx 0.533\lambda)

硬件鲁棒性检查建议：

* 第一版数值复现可先不加
* 但实验可对齐版必须补
* 至少先做 lateral shift 的随机扰动

---

## 3. 平面波假设只是理想模型

论文讨论里明确说，实验和数值不完全一致的一个原因，是训练时假设了 uniform plane wave，而真实 THz 源可能有波前畸变。

所以硬件鲁棒性里要记得：

* 你现在仿真默认的是理想平面波
* 真要往实验走时，这本身就是误差源

---

## 4. 材料参数和制造误差会影响结果

论文也指出：

* 3D 打印分辨率有限
* 折射率表征误差会造成数值与实验偏差

所以如果你以后要做更真实的版本，要检查：

* 折射率参数是否准确
* 吸收是否忽略
* 制造最小特征尺寸是否超了

---

## 5. 输出读出过程也会带来误差

实验里他们甚至对测量数据做了：

* 2×2 pixel binning 提高 SNR
* 顶部/底部 5% saturation
* 再映射到 0–1

这说明：

* 实验读出不是“直接拿仿真输出”
* 读出链本身也会改变图像统计

如果你未来做实验，不要忽略 detector / scan / binning / normalization。

---

# 最后给你一版“最短执行清单”

如果你想最实用地用，先盯这 12 条：

## 训练前

1. 明确复现数值版还是实验版
2. 明确 LR pattern 是 phase map，不是普通小图
3. 明确 pattern 尺寸、输出 FOV、传播网格三者不同
4. 开 zero padding，避免 aliasing
5. phase-only 先跑通
6. 记录初始化方式

## 训练中

7. 先用论文式 MAE，必要时再加 efficiency penalty
8. 开数据增强
9. encoder / decoder 分组学习率
10. 做单样本过拟合测试
11. 可视化整张输出平面和中心 crop
12. 明确用了哪种 normalization

## 结果分析

13. 做 1/3/5 层消融
14. 做 phase-only vs complex-valued 对照
15. 做高频 blind test，不只看字母
16. 同时看图像质量和能量效率

## 硬件鲁棒性

17. 做 16/8/6/4/2 bit quantization 测试
18. 做 misalignment vaccination 或至少 shift test
19. 记住平面波、材料、制造误差都是潜在 gap 来源
