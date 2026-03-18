# Stage 5 Protocol Freeze

## 1. 文档目的

本文档用于冻结 Stage 5（论文设定对齐）的实现边界，约束后续实现 issue 不再在以下问题上漂移：

- Stage 5 到底要对齐什么
- Stage 5 明确不包含什么
- Stage 3 / Stage 4 哪些事实已经继承且不得重写
- 当前 Stage 5 采用哪些 paper-aligned 默认值
- 哪些参数仍未完全拍板
- 每个未决项只能由哪个后续 issue 最终拍板

本文档直接约束以下 issue 家族：

- 5.2 dataset
- 5.3 optics config
- 5.4 encoder
- 5.5 loss
- 5.6 trainer
- 5.7 regular eval
- 5.8 blind eval hook
- 5.9 smoke run
- 5.10 main run

范围冲突处理规则：

- 当 `docs/paper/paper_notes.md` 记录的是更宽的复现路线，而 `docs/plan/stage_plan/stage5/stage5_plan.md` / `docs/plan/stage_plan/stage5/stage5_issue_plan.md` 记录的是当前 Stage 5 排期时，以 Stage 5 planning docs 为准。
- 当后续实现 issue 需要为未决项做最终拍板时，必须遵守本文档的 owner 约束；不允许在代码里静默拍板。

---

## 2. Stage 5 范围边界

### 2.1 In scope

Stage 5 的目标是把论文主线设定落地为可运行、可审计、可复盘的工程协议，具体包括：

- 96×96 EMNIST display 数据协议
- paper-aligned optics geometry 与 distance schedule
- phase-only encoder 到 `phi_lr` 的主线实现
- normalized MAE + efficiency term 的主线 loss
- paper-aligned training hyperparameters 与 configs
- PSNR / SSIM、bicubic baseline、blind line-pair test 的评估挂接
- paper-aligned smoke / main run 的启动路径与工件协议

### 2.2 Out of scope

以下内容明确不属于 Stage 5，仍归 Stage 6：

- quantization sweep
- misalignment / robustness
- systematic `L=1/3/5` ablations
- complex-valued / amplitude-only comparisons
- efficiency penalty 的系统化 ablation
- 大规模调参矩阵或广义框架重构

### 2.3 Stage 5 的一句话边界

Stage 5 是本仓库第一次对齐论文完整主设定的阶段；它要建立 launch-ready 的 paper-aligned pipeline，而不是提前吞进 Stage 6 的系统化实验工作。

---

## 3. 继承且冻结的事实

以下内容从 Stage 3 / Stage 4 继承，在 Stage 5 中视为冻结约束：

1. 冻结 optical contract：

```text
phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi
```

2. Stage 4 learnability gate 已通过。
3. Stage 4 artifacts 是 regression baseline，不是 Stage 5 要重做的工作。
4. Stage 5 必须在现有 optical core 之上对齐论文设定，而不是借机重写 optical core。
5. `phi_lr -> U0` 仍由 optical decoder 负责；encoder / wrapper 只负责产出 phase-domain `phi_lr`。
6. `U_out_full` 与 `I_out_full` 继续保留为调试工件；主监督 / 主评估读出仍是 `I_out_roi`。
7. Stage 5 需要支持 `L ∈ {1, 3, 5}` 的 paper-aligned config，但不把系统化深度对比挪到本阶段。

明确禁止：

- 重定义 optical tensor contract
- 在 optical decoder 后追加 electrical post-head 作为 Stage 5 主线
- 以“为了 paper alignment”为理由重写 Stage 4 已通过的 learnability 结论

---

## 4. Paper-Aligned 目标协议

本节给出 Stage 5 默认采用的 paper-aligned 目标协议。若某项仍有论文歧义，则冻结“当前默认处理 + 后续 owner”，而不是伪装成已完全确定事实。

### 4.1 Dataset protocol

- 数据源：EMNIST letters
- 原始样本：`28×28`
- 预处理：bicubic 插值到 `32×32`
- display 目标：构造 `96×96` 图像
- 数据量目标：train `60,000` / val `6,000` / test `6,000`
- 内容分布目标：
  - train / val：每张图含 `1~4` 个 letters
  - test：每张图含 `6~9` 个 letters
- 增强目标：
  - rotations：`0 / 90 / 180 / 270`
  - random flip
  - random contrast adjustment

冻结说明：

- Stage 5 dataset path 只服务论文主线 display protocol，不引入自然图像数据或 Stage 6 robustness 扰动。
- `96×96` display 的精确 tile 规则仍有实现层面的未决细节，见第 5 节 ledger。

### 4.2 Optics geometry targets

- sampling period：`0.533 λ`
- diffractive layer size：`200×200`
- propagation grid：`400×400`（zero padding）
- input / output FOV target：`96×96`
- depth：`L = 1 / 3 / 5`
- 初始化：diffractive phase masks 全零

当前 Stage 5 默认 distance mapping 采用：

- `input_to_first = d1`
- `inter_layer = d2`，在所有相邻 diffractive layers 之间重复
- `last_to_sensor = d3`

按此默认可写为：

- `L=1`：
  - `input_to_first = 6.667 λ`
  - `last_to_sensor = 173.333 λ`
- `L=3`：
  - `input_to_first = 4 λ`
  - `inter_layer = 53.334 λ`
  - `last_to_sensor = 53.334 λ`
- `L=5`：
  - `input_to_first = 2.667 λ`
  - `inter_layer = 66.667 λ`
  - `last_to_sensor = 80 λ`

冻结说明：

- 这是 Stage 5 当前默认映射，不等价于“论文已无歧义地给出了代码级 distance list”。
- 5.3 可以把该映射最终落实为 config 字段或显式距离列表，但不得在 optics core 中静默藏规则。

### 4.3 Encoder / phase representation target

- 上游输入：单通道 `96×96` HR target image
- encoder 输出：phase-only `phi_lr`
- `phi_lr` 继续遵守已有 optical contract，不改为 field-domain tensor
- optical decoder 继续负责 `phi_lr -> U0`
- 相位范围映射必须显式存在于 encoder / wrapper，而不是隐式藏在 optics core

当前 Stage 5 默认：

- `phi_lr` 目标空间尺寸先按 `32×32` 处理
- `phase_range` 默认按 `[0, 2π)` 处理

冻结说明：

- 上述两项都是 Stage 5 当前默认，不是“论文唯一明确事实”。
- 5.4 可以最终拍板 encoder 暴露的 `phi_lr` 尺寸和 phase mapping range，但在拍板前必须保持可配置。

### 4.4 Loss / gamma policy

Stage 5 目标 loss 为论文主线：

\[
\mathcal L = \frac{1}{N}\sum_i |y_i - \sigma \hat y_i| + \gamma e^{-\eta}
\]

其中：

\[
\sigma = \frac{\sum_i y_i}{\sum_i \hat y_i + \epsilon}
\]

\[
\eta = 100 \times \frac{P_o}{P_i}
\]

冻结要求：

- 主监督对象仍为 `I_out_roi`
- `sigma` 与 `eta` 的计算口径必须显式记录
- `gamma` 必须按 depth 可配置，不能硬编码为不可追踪常数

当前 Stage 5 默认 gamma policy：

- `L=1`: `gamma = 0.005`
- `L=3`: `gamma = 0.015`
- `L=5`: `gamma = 0.0`（暂定默认，非最终定论）

当前 Stage 5 默认 sigma policy：

- 先按单样本 ROI 计算 `sigma`
- 再在 batch 维聚合 loss

冻结说明：

- 5.5 可以最终拍板 `L=5 gamma` 和 `sigma` granularity。
- 在 5.5 之前，trainer / smoke / main run 只能消费这些默认值，不能自行改写 loss 语义。

### 4.5 Training hyperparameter targets

- optimizer：Adam
- decoder LR：`0.001`
- encoder LR：`0.0005`
- batch size：`40`
- epoch target：`500`
- 训练实现必须支持 encoder / decoder parameter groups

冻结说明：

- 这些是 Stage 5 的 main-run target，不等价于每个 smoke run 都必须跑满 `500 epochs`。
- 5.9 可以为了 smoke 把训练预算降到短跑版本，但必须显式标为 smoke config，不能伪装成 paper-aligned full-budget run。

### 4.6 Eval target protocol

- 常规指标：PSNR / SSIM
- baseline：bicubic，且需 anti-aliasing
- blind eval：line-pair / resolution target
- train / val / test 口径必须与 dataset protocol 对齐
- 输出 crop / normalization 口径必须显式记录

冻结说明：

- 5.7 负责 regular eval + bicubic baseline
- 5.8 负责 blind line-pair hook
- eval 路径不得反向修改 optics crop 语义；crop/FOV 对齐由第 5 节 ledger 约束

---

## 5. Unresolved Parameters Ledger

下表冻结的是“当前默认处理 + owner issue + 不允许偷拍板的 issue”。

| 未决项 | 当前 Stage 5 默认处理 | 仍未知的点 | 允许最终拍板的 issue | 必须不得静默拍板的 issue | 配置要求 |
| --- | --- | --- | --- | --- | --- |
| `phi_lr` size | 默认按 `32×32`；encoder 输出尺寸与 optics `input_pattern_hw` 必须一致 | 论文是否足以把 `32×32` 视为唯一代码级尺寸；是否需要额外显式说明与 `3×` SR 的关系 | `5.4` | `5.2`, `5.5`, `5.6`, `5.7`, `5.8`, `5.9`, `5.10` | 在 `5.4` 前必须保持可配置 |
| `96×96` display tiling rule | 默认按 `3×3` 个 `32×32` cells 构造，occupied cell 数量遵守 train/val/test 分布；样本预览必须落地 | cell 采样、空位策略、是否允许重复字母、随机种子与布局是否固定 | `5.2` | `5.4`, `5.5`, `5.6`, `5.7`, `5.8`, `5.9`, `5.10` | tile 规则、seed、split 口径必须进 config / preview |
| distance mapping 到 `input_to_first / inter_layer / last_to_sensor` | 默认按 `d1 / d2 / d3` 对应；`d2` 在多层间重复；`L=1` 无 `inter_layer` | 论文记号是否无歧义地支持该重复规则；是否需要 depth-specific distance list 表达 | `5.3` | `5.4`, `5.5`, `5.6`, `5.7`, `5.8`, `5.9`, `5.10` | 必须显式进 optics config，不得埋在实现分支里 |
| `L=5` gamma policy | 当前默认 `gamma = 0.0`，并要求 `gamma_by_depth` 可配置 | 论文“其它设计 γ=0”是否应直接覆盖 `L=5` 主线 run | `5.5` | `5.6`, `5.7`, `5.8`, `5.9`, `5.10` | `gamma_by_depth` 必须可配置并写入 summary |
| phase mapping range | 当前默认 `[0, 2π)`；映射在 encoder / wrapper 内完成 | `[0, 2π)` 与 `[-π, π]` 哪个更接近论文 / 器件语义 | `5.4` | `5.3`, `5.5`, `5.6`, `5.9`, `5.10` | 必须保留显式 `phase_range` 配置 |
| sigma normalization granularity | 当前默认“逐样本 ROI 求 `sigma`，再 batch 聚合” | 论文公式在 mini-batch 实现中是否应按逐 batch 或逐样本解释 | `5.5` | `5.6`, `5.7`, `5.8`, `5.9`, `5.10` | 必须保留显式 `sigma_mode` 或等价记录字段 |
| output crop / FOV alignment details | 当前默认延续 Stage 3/4 center-crop 语义；目标 crop 为 `96×96`，无额外 margin | full `400×400` grid 到 `96×96` FOV 的精确对齐、像素中心、offset / margin 是否需要更严格说明 | `5.3` | `5.6`, `5.7`, `5.8`, `5.9`, `5.10` | crop size、origin、offset 必须显式记录 |

Ledger 约束：

- owner issue 若要把默认值改成最终值，必须在其交付物中显式记录“相对本文档默认的变化”。
- 非 owner issue 只能消费当前默认或配置项，不能用代码行为偷拍板。
- 5.9 / 5.10 只能验证冻结后的协议，不是拍板协议的地方。

---

## 6. 对后续 Issue 家族的约束

### 6.1 Dataset（5.2）

- 可以最终拍板：`96×96` tiling rule、split 生成细则、augmentation 参数暴露方式
- 必须保持：样本预览、split 口径、随机种子可追踪
- 不得触碰：optics geometry、phase mapping、loss 语义、eval crop

### 6.2 Optics config（5.3）

- 可以最终拍板：distance mapping、paper-aligned grid/padding、output crop / FOV alignment 细节
- 必须保持：冻结 optical contract 不变；no optical-core redesign
- 不得触碰：dataset tile 规则、encoder 内部结构、loss 公式含义

### 6.3 Encoder（5.4）

- 可以最终拍板：`phi_lr` 暴露尺寸、phase mapping range
- 必须保持：输出对象仍是 phase-domain `phi_lr`；`phi_lr -> U0` 仍在 optical decoder
- 不得触碰：distance mapping、loss 公式、eval 口径

### 6.4 Loss（5.5）

- 可以最终拍板：`L=5 gamma`、`sigma` granularity、efficiency-term 配置接口
- 必须保持：gamma 按 depth 可配置；loss 行为可追踪
- 不得触碰：dataset 构造、optics crop 几何、trainer 的 smoke/main 预算策略

### 6.5 Trainer（5.6）

- 必须消费已冻结 dataset / optics / encoder / loss config
- 可以新增：resume、artifact 保存、paper-aligned config 入口
- 不得拍板：tiling、distance mapping、phase range、gamma policy、sigma granularity、crop 语义

### 6.6 Eval（5.7 / 5.8）

- 5.7 只负责 regular eval + bicubic baseline
- 5.8 只负责 blind line-pair hook
- 二者都必须复用已冻结 crop / normalization 口径
- 不得反向修改：dataset split、optics geometry、loss 默认值

### 6.7 Smoke / main run（5.9 / 5.10）

- 5.9 只验证 pipeline 稳定性与工件完整性，可使用缩短预算，但必须显式标记为 smoke
- 5.10 负责 paper-aligned main-run 启动与结果记录
- 二者都不得作为协议拍板入口；若发现协议错误，只能回写文档/issue 再修正，不得靠 run 脚本悄悄改默认

---

## 7. Acceptance / Exit Criteria

### 7.1 Stage 5 engineering-complete 的最低标准

在工程层面，Stage 5 视为完成至少需要同时满足：

1. `stage5_protocol_freeze.md` 已存在，且未决项 owner 明确。
2. paper-aligned dataset path 已落地，并有可复盘的 preview / split 记录。
3. `L=1/3/5` 的 paper-aligned optics configs 已落地，forward sanity 可通过。
4. paper-aligned encoder、loss、trainer、regular eval、blind eval hook 已落地。
5. 关键未决项在 owner issue 拍板前都保持可配置并被日志记录。
6. 至少一次 paper-aligned smoke run 已完成，且工件完整。
7. main-run recipe 已清楚：config、命令、工件路径、评估入口都可审计。

### 7.2 更强但资源相关的证据

以下属于更强证据，但受算力 / 时间预算影响，不是前置工程完成门槛：

- 一次真实启动并有结果记录的 paper-aligned `L=5` main run
- val / test 上可复盘的 PSNR / SSIM
- blind line-pair 结果与 bicubic baseline 的对照记录
- 明确区分“代码 / 配置已就绪”与“长训预算是否跑满”

### 7.3 Stage 6 入口边界

只有当 Stage 5 已具备可信的 paper-aligned launch path 之后，以下工作才允许进入 Stage 6：

- quantization sweep
- misalignment / robustness
- systematic `L=1/3/5` ablations
- complex-valued / amplitude-only comparisons

在此之前，不允许以后续实验名义反向修改本文档定义的 Stage 5 边界。
