# Stage 4 Protocol Freeze

## 1. 文档目的

本文档用于冻结 Stage 4 的最小闭环训练协议与 tensor contract，目标是让后续实现 issue 在以下问题上不再反复解释：

- Stage 4 到底要证明什么
- 最小闭环的系统边界是什么
- encoder 与 optical decoder 的责任如何切分
- 首版训练、读出、验收应以什么口径执行

本文档不是实现说明书，也不是 Stage 5/6 计划草案。

当不同资料存在冲突时，本文件采用以下优先级：

1. `docs/ai/PROJECT_CONTEXT.md`
2. `docs/plan/plan_overview.md`
3. `docs/plan/stage3_contract_freeze.md`
4. 当前 `src/` 与 `scripts/` 实现
5. `docs/paper/paper_notes.md`
6. `docs/execution/results_summary.md`
7. `docs/execution/experiment_log.md`

本文中内容分三类表达：

- 已冻结事实：当前仓库与 Stage 3 已经确认的边界
- Stage 4 首版选择：为了尽快验证“能否学”而故意采用的简化协议
- `待确认`：当前仓库没有足够证据冻结的实现细节

---

## 2. Stage 4 核心问题

Stage 4 只回答一个问题：

> 在不重写既有 optical core 的前提下，把最小 encoder 接到已冻结的 diffractive decoder 后，系统能否在梯度驱动下完成端到端学习？

这里的“系统”固定指：

- 最小 encoder
- 既有 optical decoder
- 直接 optical ROI readout
- 基于 `I_out_roi` 的监督

Stage 4 的通过标准不是论文最终指标，而是以下最小证据已经出现：

- 端到端 forward / backward 稳定
- 梯度能到达 encoder 与 optical decoder 的可训练参数
- 单样本过拟合可行
- 真实小子集闭环训练可行

---

## 3. Stage 4 范围边界

### 3.1 In scope

- 复用 Stage 3 已冻结 optical contract
- 定义最小 encoder 的输入输出边界
- 建立 `HR input -> encoder -> phi_lr -> optical decoder -> I_out_roi -> loss` 的首版闭环
- 建立首版 Stage 4 trainer / artifact / acceptance protocol
- 在真实数据上完成单样本过拟合与小子集闭环训练

### 3.2 Out of scope

- Stage 5 的论文 full-setting 对齐
- Stage 6 的系统化消融
- blind line-pair 正式评估
- 相位量化 sweep
- misalignment / hardware robustness
- `L=1/3/5` 的系统化对比实验
- efficiency penalty 的系统化开关比较
- 大规模训练、长周期训练、正式 benchmark

### 3.3 边界原则

1. Stage 4 必须建立在现有 optical core 之上，不能借机重定义 `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`。
2. Stage 4 首版要先证明“能学”，而不是先追论文最终设置。
3. Stage 3 的 synthetic / toy verification 只能作为前置信心，不能充当 Stage 4 通过证据。
4. 若某个实现选择当前仓库尚无足够依据，必须显式标为 `待确认`，不能伪装成论文已明确。

---

## 4. 当前仓库状态摘要

### 4.1 已可复用资产

- `src/models/optics/diffractive_decoder.py`
  - 已实现冻结链路 `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
  - 已提供 `forward_from_phase(...)`、`forward_from_field(...)`、`forward_from_phase_provider(...)`
  - 已固定 `L` 的语义为 trainable diffractive phase masks 数量，当前支持 `L ∈ {1, 3, 5}`
- `src/models/optics/phase_provider.py`
  - 已定义 `PhaseProvider` contract：`upstream_input -> phi_lr`
  - 已把“谁产出 phase”和“谁执行 phase-to-field”分离
- `scripts/train_electronic_baseline.py`
  - 已提供可信的训练骨架参考：数据加载、train/val split、单样本 overfit 模式、checkpoint、`summary.json`、`metrics.json`、sample 图保存
- `src/models/electronic_baseline.py`
  - 已提供 `HR -> latent -> recon` 的最小电子闭环参考
  - 其中 encoder 下采样路径可作为 Stage 4 最小 encoder 的概念锚点，但不是可直接复用的 optical encoder
- `src/eval/evaluator.py`
  - 已提供统一 `PSNR / SSIM` 评估入口，可作为 Stage 4 辅助指标接口参考

### 4.2 当前仍缺失的基础设施

- 真正的 Stage 4 hybrid wrapper
- 真正的 Stage 4 最小 encoder 模块
- 真正的 Stage 4 trainer
- 真正的 Stage 4 eval runner
- Stage 4 optics config / training config
- 清晰的 Stage 4 数据模块与自动化测试入口

当前仓库中的 `scripts/train.py`、`scripts/eval.py`、`scripts/visualize.py` 仍为空文件，`configs/` 中也尚无 Stage 4 专用配置。

### 4.3 当前哪些验证是可信的

- Stage 3 optical contract 已落地，readout / crop / depth forward sanity 已通过
- decoder-only single-sample fitting 已显示 loss 可下降
- decoder-only small-subset capacity check 已显示 `L=1/3/5` 在 tiny protocol 下都可优化
- 电子 baseline 路径已经通过单样本 overfit 与小子集训练 sanity，可作为 Stage 4 训练骨架的可信参考

### 4.4 当前哪些还只是初步 toy / synthetic verification

- `scripts/train_decoder_only_small_subset.py` 使用 synthetic ROI targets
- 该脚本使用 toy grid 与 crop 尺寸：
  - `input_pattern_hw = (24, 24)`
  - `layer_hw = (32, 32)`
  - `propagation_hw = (48, 48)`
  - `output_crop_hw = (20, 20)`
- 这些结果只说明 optical decoder 在 toy 协议下可学，不等价于真实 Stage 4 端到端训练已经成立

---

## 5. 已冻结的 Stage 3 optical contract

以下内容在 Stage 4 中视为继承约束，不得重写：

1. 主链路固定为：

```text
phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi
```

2. `forward_from_phase(...)` 是 Stage 4 主线入口：
   - 输入为 phase-domain tensor `phi_lr`
   - optical decoder 负责 `phi_lr -> U0` 的 phase-to-field 构造
3. `forward_from_phase_provider(...)` 是允许的集成边界：
   - provider 负责产出 `phi_lr`
   - optical decoder 仍负责 phase-to-field 与后续传播
4. `forward_from_field(...)` 仅用于 field-domain 调试与验证，不是 Stage 4 主训练入口
5. full-grid forward 与 ROI supervision 必须显式分离：
   - `U_out_full` 与 `I_out_full` 必须保留用于诊断
   - 主监督对象是 `I_out_roi`
6. optical module 只负责 direct optical readout，不包含 electrical decoder / refinement head
7. 当前支持的 diffractive depth 仍为 `L ∈ {1, 3, 5}`

---

## 6. Stage 4 最小闭环定义

Stage 4 首版最小闭环固定为：

```text
HR input -> minimal encoder -> phi_lr -> optical decoder -> I_out_roi -> loss
```

更具体地说：

1. 上游输入是单通道 HR 图像 tensor。
2. 最小 encoder 的唯一核心职责，是把 HR 输入映射为可被 optical decoder 消费的 `phi_lr`。
3. optical decoder 继续使用现有 Stage 3 core，不新增 electrical post-head。
4. 首版训练直接以 `I_out_roi` 为读出与监督对象。
5. 首版 Stage 4 只接受真实数据闭环，不接受 synthetic target 充当验收样本。

Stage 4 首版选择如下：

- 训练形态：joint training
  - encoder 与 optical decoder 默认同时参与训练
  - 若为调试临时冻结 decoder，必须在产物中明确标注；此类结果不计入 Stage 4 正式验收
- 数据形态：优先沿用当前电子 baseline 已验证的数据读取与小规模运行方式
- 系统形态：不增加 optical decoder 之后的电子重建头

`待确认`：

- 首版默认使用 `L=1` 还是 `L=3` 作为最先落地的 smoke depth
- 首版 Stage 4 是否直接采用与当前电子 baseline 对齐的 `96 -> 24 -> 96` 空间尺度
- 首版 Stage 4 optics config 的精确 `input_pattern_hw / propagation_hw / output_crop_hw`

---

## 7. 首版张量契约（tensor contract）

### 7.1 已冻结 contract

| 名称 | 含义 | 形状语义 | 责任边界 |
| --- | --- | --- | --- |
| `upstream_input` / `x_hr` | encoder 的上游输入；首版为单通道 HR 图像 | `[B, 1, H_hr, W_hr]` | trainer / dataset 负责提供；当前可信锚点是电子 baseline 的单通道 EMNIST 路径 |
| `target_hr` | 监督目标；首版与 `x_hr` 指向同一张 HR 目标图像 | `[B, 1, H_t, W_t]` | trainer 负责保存与记录其预处理规则 |
| `phi_lr` | encoder 输出的低分辨率 phase-domain 表示，不是通用 latent feature map | `[B, 1, H_phi, W_phi]` | encoder / phase-provider 负责产出；进入 optical decoder 前应已经是“相位语义” |
| `U0` | 由 `phi_lr` 构造出的输入复场 | `[B, 1, H_prop, W_prop]`，complex | optical decoder 负责构造；主线默认 `amplitude=None`，即单位振幅 |
| `U_out_full` | sensor plane 的 full-grid 复场 | `[B, 1, H_prop, W_prop]`，complex | optical decoder 负责输出并保留 |
| `I_out_full` | full-grid 输出强度 | `[B, 1, H_prop, W_prop]`，real nonnegative | optical decoder 负责输出并保留 |
| `I_out_roi` | 输出 ROI 强度，是首版主监督读出 | `[B, 1, H_roi, W_roi]`，real nonnegative | optical decoder 负责通过显式 center crop 得到 |
| `target_roi` | 与 `I_out_roi` 对齐后的监督目标 | `[B, 1, H_roi, W_roi]` | trainer 负责确定性对齐；该对齐规则必须写入 config / summary |

固定语义如下：

1. batch 维始终保留为 `B`。
2. channel 维首版固定为 `1`。
3. `phi_lr` 是 phase-domain tensor，`forward_from_phase(...)` 不负责把普通 feature map 解释成 phase。
4. `H_phi, W_phi` 必须等于 optical decoder 的 `input_pattern_hw`。
5. `H_prop, W_prop` 必须等于 optical decoder 的 `propagation_hw`。
6. `H_roi, W_roi` 必须等于 `readout_config.output_crop_hw`。

### 7.2 phase-domain 责任边界

以下边界在 Stage 4 中冻结：

- encoder 可以内部使用 raw activations、bounded outputs 或其他参数化方式，但它暴露给 optical decoder 的对象必须是 `phi_lr`
- 若 encoder 需要把内部 raw tensor 映射到某个 phase range，该映射属于 encoder / hybrid wrapper 责任，不属于 optical decoder
- optical decoder 对 `phi_lr` 的主线解释固定为：
  - 这是一个实值相位张量
  - decoder 负责通过 `exp(j * phi)` 构造 coherent field
  - 主线默认不额外学习 amplitude

### 7.3 已冻结与待确认的区分

已冻结：

- 输入边界是 `HR image -> phi_lr`
- `phi_lr` 是相位语义，不是 field 语义
- optical decoder 的主线输入是 `phi_lr`，主线输出是 `I_out_roi`
- 首版训练监督对象是 `I_out_roi`
- 首版系统不允许在 optical decoder 之后接电子 refinement head

`待确认`：

- `H_hr / W_hr` 的精确数值
- `H_phi / W_phi` 的精确数值
- `H_prop / W_prop` 的精确数值
- `H_roi / W_roi` 的精确数值
- encoder 内部 raw output 到最终 `phi_lr` 的具体 phase range 映射
- 首版 `target_hr -> target_roi` 的具体空间对齐实现

为减少首版歧义，Stage 4 首版建议优先选择：

- `output_crop_hw` 直接等于首版监督目标空间尺寸

这样首版实现可以避免引入额外的学习型 readout 对齐逻辑。
但该数值本身目前仍属 `待确认`，不能在本文件中伪装成已落地事实。

---

## 8. 首版训练协议

### 8.1 必须执行的两级协议

Stage 4 首版验收固定分为两步，顺序不可颠倒：

1. 单样本过拟合
2. 真实小子集闭环训练

只有当第 1 步通过后，第 2 步结果才有解释价值。

### 8.2 单样本过拟合协议

- 使用真实数据中的单一样本，不使用 synthetic target
- 使用完整闭环：
  - `x_hr -> encoder -> phi_lr -> decoder -> I_out_roi -> loss`
- 默认 joint training encoder + optical decoder
- 目标不是 generalization，而是证明：
  - loss 可下降
  - `phi_lr` 会变化
  - optical readout 会随训练变化
  - 梯度能到达 encoder 与 optics

### 8.3 小子集闭环训练协议

- 使用真实数据的小 train/val 子集
- 数据抽样、split、sample 保存方式应尽量复用 `scripts/train_electronic_baseline.py` 的工程骨架
- 训练规模保持小：
  - 小 subset
  - 小 batch
  - 短周期
- 目标不是论文指标，而是验证：
  - 训练过程稳定
  - 不止单一样本可下降
  - artifacts 足够支撑排错

### 8.4 优化与运行约定

Stage 4 首版选择：

- optimizer 首选 `Adam`
- encoder 与 optical decoder 允许使用独立 parameter groups
- 精确学习率、batch size、epoch / step 数目前均为 `待确认`
- 所有运行参数必须写入保存产物，不能只存在命令行历史

### 8.5 Artifact 保存约定

Stage 4 trainer 首版至少应保存以下产物：

- `summary.json`
  - 记录 config 快照、数据子集信息、depth、shape、loss 约定、是否 joint training、artifact 路径
- `metrics.json`
  - 记录关键训练结果摘要
- `checkpoints/`
  - 至少包含 `best` 与 `final`
- `samples/`
  - 至少包含输入 HR、`phi_lr` 可视化、`I_out_roi`、误差图

单样本过拟合额外要求至少保存一份 raw tensor snapshot，用于排查：

- `phi_lr`
- `U0`
- `I_out_full`
- `I_out_roi`
- `target_roi`

### 8.6 梯度可观测性约定

Stage 4 首版每个正式验收 run 都必须证明以下事实至少被观测并记录一次：

- encoder 参数梯度存在、有限、非全零
- optical decoder 可训练参数梯度存在、有限、非全零
- 训练过程中未出现 `NaN / Inf`

如果某次 run 仅验证了 loss 下降，但没有记录梯度可观测性，则该 run 不能作为完整 Stage 4 通过证据。

---

## 9. 首版损失与读出约定

### 9.1 首版读出约定

首版主读出固定为：

- `I_out_roi`

明确不做：

- 在 `I_out_roi` 之后追加电子 refinement head
- 用 `I_out_full` 直接做主监督
- 把 field-domain `U_out_full` 作为训练主读出

### 9.2 首版损失约定

首版主损失固定选择为 Stage 3 已验证口径的 ROI `normalized MAE`：

\[
\mathcal L_{\text{stage4-v1}} = \frac{1}{B}\sum_b \mathrm{mean} \left| y_b - \sigma_b \hat y_b \right|
\]

其中：

\[
\sigma_b = \frac{\sum y_b}{\sum \hat y_b + \epsilon}
\]

含义如下：

- `\hat y_b` 对应 `I_out_roi`
- `y_b` 对应 `target_roi`
- `\epsilon` 用于数值保护
- 归一化因子按样本独立计算，再在 batch 维求平均

### 9.3 这是刻意的 Stage 4 简化

该选择是 Stage 4 首版工程协议，不代表论文最终损失已经对齐。

明确后置到 Stage 5/6 的内容包括：

- efficiency penalty 的正式纳入
- 论文最终 loss 组合对齐
- 量化、鲁棒性或 blind test 驱动的附加损失

### 9.4 辅助指标约定

- `PSNR / SSIM` 可以作为辅助日志指标
- 但由于 Stage 4 eval runner 尚未落地，其具体归一化与裁剪口径当前仍属 `待确认`
- 因此 Stage 4 首版验收不以 `PSNR / SSIM` 达到某个阈值为条件

---

## 10. 验收标准

### 10.1 单样本过拟合通过条件

以下条件必须同时满足：

1. 使用真实数据单样本的完整闭环 run 可以稳定完成
2. `best_loss < initial_loss`
3. `phi_lr` 与 `I_out_roi` 在训练前后确实发生变化，且不是常数塌缩
4. encoder 与 optical decoder 的梯度都被观测到且为有限非零
5. `summary.json`、`metrics.json`、checkpoint、sample 图、raw tensor snapshot 保存完整

### 10.2 小子集闭环训练通过条件

以下条件必须同时满足：

1. 使用真实数据小子集，而非 synthetic target
2. 训练 run 可稳定完成，不出现 `NaN / Inf`
3. 聚合训练损失相对初始状态出现下降
4. 至少能保存多样本的 `phi_lr / I_out_roi / 误差图` 预览
5. 训练产物足够复盘：
   - config / summary
   - checkpoints
   - metrics
   - sample 可视化
   - 梯度可观测性记录

### 10.3 进入 Stage 5 前必须具备的证据

进入 Stage 5 前，至少必须已有以下证据同时成立：

1. 一次通过的真实单样本 overfit 闭环 run
2. 一次通过的真实小子集闭环 run
3. 两类 run 都遵守本文档的同一 tensor contract
4. 结果不依赖 synthetic target、decoder-only 路径或额外电子后处理头
5. 所有 `待确认` 的 Stage 4 首版实现选择都已在 config / summary 中被明确记录

如果以上证据缺失，则不应进入论文 full-setting 对齐。

---

## 11. 非目标与延后事项

以下内容明确不属于本文件冻结的 Stage 4 首版目标：

- paper-exact full setting alignment
- blind line-pair 正式盲测
- quantization sweep
- misalignment robustness
- `L=1/3/5` 的系统化比较
- efficiency penalty 的系统化 ablation
- large-scale training
- complex-valued / amplitude-only 输入分支
- THz 硬件设定或 material model 对齐

这些事项应后置到 Stage 5/6 或更后阶段，不得反向挤入 Stage 4 首版验收。

---

## 12. 对后续 issue 的约束

### 12.1 对最小 encoder issue 的约束

- 只能决定 encoder 内部拓扑，不能重定义 optical contract
- 输出必须是 `phi_lr`，而不是 field tensor 或 generic latent tensor
- 若内部存在 raw-to-phase 映射，该映射必须在 encoder / wrapper 内部完成

### 12.2 对 hybrid wrapper issue 的约束

- 必须复用 `forward_from_phase(...)` 或 `forward_from_phase_provider(...)`
- 不允许把 `phi_lr -> U0` 逻辑散落到 trainer 中
- 不允许绕开 `I_out_roi`，直接引入新的 readout 主链路

### 12.3 对 Stage 4 trainer issue 的约束

- 必须同时支持：
  - 单样本 overfit 模式
  - 小子集闭环训练模式
- 必须保存 `summary.json`、`metrics.json`、`checkpoints/`、`samples/`
- 必须记录梯度可观测性

### 12.4 对单样本 overfit issue 的约束

- 必须使用真实数据样本
- 必须走完整闭环
- decoder 临时冻结只能用于调试，不能作为通过结论

### 12.5 对小子集 run issue 的约束

- 必须使用真实数据小子集
- 必须保持“小而可排错”的 Stage 4 范围
- 不得把该 issue 扩展成 Stage 5 论文指标追逐或大规模训练

### 12.6 对后续扩展 issue 的总约束

在本文档定义的最小闭环证据补齐之前，以下类型的 issue 默认不应启动：

- 论文 full-setting 对齐
- blind line-pair 自动化
- quantization / robustness sweep
- 深度系统化比较
- 大规模训练与系统化消融
