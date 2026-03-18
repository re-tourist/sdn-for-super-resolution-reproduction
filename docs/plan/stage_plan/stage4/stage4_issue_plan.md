**GPT-5.3**

下面给出一份 **Stage 4 详细工程推进计划**，目标是能直接拆成 GitHub issues 去执行。内容严格基于你给的仓库现状，而不是空泛地讲“做联合训练”。目前最关键的事实有三点：

第一，项目已经明确进入 **Stage 4: minimal closed-loop training**，核心问题不是追论文指标，而是验证“接入 encoder 后系统是否还能学”。这一点在 `PROJECT_CONTEXT.md` 和 `plan_overview.md` 里都已经拍板。 

第二，**光学 contract 已冻结**，Stage 4 必须承接现有链路
`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`，不能重写 optical core。当前 `DiffractiveDecoder` 也已经把 `forward_from_phase(...)`、`forward_from_field(...)`、`forward_from_phase_provider(...)` 这些 Stage 4 接口钩子留好了。 

第三，当前 repo **还没有真正的 Stage 4 trainer**；Stage 3 的训练验证脚本仍然是 **synthetic target / toy grid** 的 capacity check，例如 `train_decoder_only_small_subset.py` 使用的是 `24/32/48/20` 这套 toy 尺寸和 synthetic target library，而不是 end-to-end 数据闭环。 

基于这些前提，Stage 4 不能一口气铺开，而要按“**先把闭环接上，再验证梯度，再做最小训练，再补评估与文档**”的节奏推进。

---

# 一、Stage 4 的工程目标重新落地

可以把 Stage 4 压缩成一句工程话：

> **在不改动已冻结 optical contract 的前提下，新增一个最小 encoder、一个最小 Stage 4 trainer、一个最小数据闭环协议，并通过单样本 overfit 和小数据集训练证明系统可学习。**

这里的“可学习”不是一句口号，而是至少要看到四个证据：

1. loss 稳定下降
2. encoder 参数收到非零梯度
3. optical 参数收到非零梯度
4. `phi_lr / I_out_roi / recon-vs-target` 的可视化能说明模型不是瞎抖动，而是在朝目标收敛

---

# 二、先把当前 repo 里“能复用的”与“必须新开发的”分开

## 2.1 可直接复用的部分

### A. Optical core

直接复用 `DiffractiveDecoder`，尤其是：

* `forward_from_phase(...)`
* `build_input_field(...)`
* `readout_from_field(...)`
* `forward_from_phase_provider(...)`

这意味着 Stage 4 的主工作不是“怎么再写一个 optical decoder”，而是“怎么把 encoder 输出安全地接到 `phi_lr` 上”。

### B. 可信训练骨架

可以直接参考 `train_electronic_baseline.py` 的这些工程套路：

* argparse + config merge
* output/checkpoint/sample 目录组织
* train/val loop
* metrics.json / summary.json 输出
* 单样本 overfit 开关
* 中间样本可视化保存

它不是 Stage 4 trainer，但它已经是一套可信骨架。

### C. Stage 3 optical learnability 经验

`train_decoder_only_small_subset.py` 已经证明：

* optical 参数可以被优化
* ROI loss 可以下降
* synthetic target 上 decoder 有容量

但这只是 **decoder-only + synthetic target**，不能拿来当 Stage 4 完成。

---

## 2.2 必须新开发的部分

### A. Stage 4 最小 encoder

当前 repo 没有一个明确的“给 optical decoder 喂 `phi_lr` 的最小 learned encoder”主线实现。

### B. Stage 4 通用 trainer

`PROJECT_CONTEXT` 已明确写了：

* `scripts/train.py` 空
* `scripts/eval.py` 空
* `scripts/visualize.py` 空
* 还没有 real Stage 4 general trainer / eval runner / optics config 

### C. Stage 4 数据协议

Stage 4 不一定立刻对齐论文 96×96 full setting，但必须定义清楚：

* 输入 HR 是什么尺寸
* encoder 输出 `phi_lr` 是什么尺寸
* optical toy grid 是否继续沿用
* target 是 `I_out_roi` 对谁监督

### D. 中间产物与验收协议

Stage 4 必须把“能不能学”变成可检查对象，而不是口头判断。

---

# 三、Stage 4 建议拆成 8 个 GitHub Issues

我建议 Stage 4 拆成 **8 个 issues**。这样颗粒度够细，便于你和 Codex 分轮推进，也便于在 PR / commit 里追踪。

建议命名方式：

* Milestone: `Stage 4 — Minimal Closed-Loop Training`
* Issue 编号格式：`4.1 ~ 4.8`

---

# Issue 4.1 — Freeze Stage-4 minimal protocol and tensor contract

## 这个 issue 要解决什么

先把 Stage 4 的边界拍死，避免 coder 一上来写大而空的 trainer。

## 为什么必须先做

你现在最容易犯的错，不是代码不会写，而是 **Stage 4 和 Stage 5 混掉**。
必须先把以下问题写成文档并冻结：

* Stage 4 只验证 learnability，不追论文 full-setting
* 继续承接 Stage 3 optical contract
* 先用 toy optical grid，不重写 optical core
* 先用最小数据闭环，不做大规模 sweep

## 输入

* `docs/ai/PROJECT_CONTEXT.md` 
* `docs/plan/plan_overview.md` 
* `src/models/optics/diffractive_decoder.py` 
* `scripts/train_decoder_only_small_subset.py` 

## 产物

新增文档，建议路径：

`docs/plan/stage4_protocol_freeze.md`

## 文档里要冻结的内容

至少写清：

* Stage 4 输入 / 输出链路
* `encoder -> phi_lr -> optical decoder -> I_out_roi -> loss`
* `phi_lr` 的 shape / range 约束
* 首版是否沿用 toy grid（建议是）
* 首版 target 是否直接监督 ROI（建议是）
* 首版 loss 先用 normalized MAE（建议是）
* 首版实验协议：单样本 overfit + 小数据集闭环训练

## 验收标准

文档能回答以下问题：

* Stage 4 和 Stage 5 的边界在哪里
* `phi_lr` 的 contract 到底是什么
* toy optical setup 是否继续复用
* 首轮训练是监督 `I_out_roi` 还是别的量

## 风险点

最大风险是把 Stage 4 写成“论文设定预演”。

## GitHub issue 正文建议

Title:
`Freeze Stage 4 minimal protocol and tensor contract`

Body:

```md
## Background
Stage 4 should validate end-to-end learnability after connecting the encoder to the frozen optical decoder contract, without jumping to Stage 5 paper-aligned full settings.

## Goal
Write and freeze a Stage 4 protocol document that defines:
- minimal closed-loop scope
- tensor contracts
- first training scale
- first loss/readout choice
- acceptance checks

## Tasks
- Review current PROJECT_CONTEXT, plan_overview, diffractive_decoder, and Stage 3 toy scripts
- Define the Stage 4 dataflow and tensor contract
- Freeze the initial loss/readout choice
- Freeze the first minimal experiment protocol
- Document non-goals explicitly

## Deliverable
- `docs/plan/stage4_protocol_freeze.md`

## Acceptance
The document must make it impossible to confuse Stage 4 with Stage 5.
```

---

# Issue 4.2 — Implement minimal Stage-4 encoder that outputs phase tensor

## 这个 issue 要解决什么

实现一个最小 encoder，把上游 HR 图像映射成 `phi_lr`。

## 关键判断

这里不要急着按论文 encoder 结构复现。`paper_notes.md` 里记录了论文 encoder 大致结构和学习率量级，但 Stage 4 目标不是 strict paper alignment。论文最终 encoder 可放到 Stage 5。

## 设计建议

首版 encoder 应满足：

* 输入：HR image
* 输出：单通道 `phi_lr`
* 输出 shape：与 `input_pattern_hw` 对齐
* 输出 range：显式约束到 phase 域，例如 `[-pi, pi]`

建议新文件：

`src/models/stage4/minimal_phase_encoder.py`

建议结构：

* 2~4 个 conv blocks
* 最后一个 conv 输出 1 channel
* `phi_lr = π * tanh(raw)` 或其他明确 phase mapping

这里建议用 `π * tanh(raw)`，理由：

* 与当前 Stage 3 toy 脚本中对 phase raw 做约束的思路一致。`train_decoder_only_small_subset.py` 里是 `math.pi * torch.tanh(phase_raw)`。
* 比直接线性输出更稳
* 不会把 phase 输出 contract 留成黑盒

## 输入

* `src/models/optics/diffractive_decoder.py` 
* `scripts/train_decoder_only_small_subset.py` 

## 产物

* `src/models/stage4/minimal_phase_encoder.py`
* 可选：`src/models/stage4/__init__.py`

## 验收标准

* forward 输出 shape 正确
* 输出值域满足 phase contract
* 单独跑一次 fake input 没有 NaN / Inf
* 能接入 `decoder.forward_from_phase(...)`

## 风险点

* 输出 spatial size 不匹配
* 过强 saturating 映射导致前期梯度过小
* coder 擅自把 optical core 改掉

## GitHub issue 正文建议

Title:
`Implement minimal phase encoder for Stage 4 closed-loop training`

Body:

```md
## Background
Stage 4 needs a minimal learned encoder that outputs a phase tensor compatible with the frozen optical contract:
`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`.

## Goal
Add a small encoder module that maps an HR input image to a single-channel phase tensor with explicit range control.

## Tasks
- Create `src/models/stage4/minimal_phase_encoder.py`
- Make the output shape match the Stage 4 chosen optical input pattern size
- Apply an explicit phase-range mapping
- Keep the implementation minimal and debuggable
- Do not modify the optical core contract

## Deliverable
- Minimal phase encoder module
- Small self-check / smoke-run example if needed

## Acceptance
- Encoder output shape matches decoder input expectations
- Output range is explicit and bounded
- Forward pass works with the current diffractive decoder
```

---

# Issue 4.3 — Build Stage-4 minimal dataset path and target adapter

## 这个 issue 要解决什么

把 Stage 4 的训练输入、监督目标、尺寸变换路径固定下来。

## 为什么单独拆一个 issue

因为你现在最大的灰区之一是：

> Stage 4 到底用什么数据形态训练？

当前 `train_electronic_baseline.py` 是电子自编码重建路径，输入输出都是 HR 图像；而 Stage 3 optical toy 脚本则直接拟合 synthetic ROI。 

Stage 4 必须新增一个 **adapter layer**，把真实图像数据接到 optical ROI 监督上。

## 建议方案

首版先做一个最小、诚实、可跑通的协议：

* 数据源：EMNIST（复用已有可靠下载与 transform 逻辑）
* 首版 target：与 `output_crop_hw` 同尺寸的 target ROI
* 处理方式：从 HR 图像生成与 ROI 对齐的 target
* 先不追论文 96×96 display dataset；那是 Stage 5 的事情。论文里的 96×96 多字母拼图数据协议先记账，不在此 issue 强行对齐。

具体可选两条路：

### 路线 A（更稳）

* 输入原始单字母图像
* resize 成 Stage 4 HR input
* 再 resize/crop 成 ROI target

这更像“最小 learnability 闭环”。

### 路线 B（更贴近未来）

* 先构造轻量版 tiled display sample
* 再生成 ROI target

但这会把 Stage 4 做重。

我建议 **先走路线 A**。

## 新文件建议

* `src/data/stage4_emnist.py`
* 或更轻量一点：`src/data/stage4_dataset.py`

## 产物

* 数据集封装
* target adapter
* 一个 batch 可视化检查脚本或函数

## 验收标准

* 一个 batch 能输出：`input_hr`, `target_roi`
* `target_roi` 尺寸与 `I_out_roi` 对齐
* 保存 4~8 个样本图，肉眼可看

## 风险点

* 监督目标与 optical readout 不对齐
* resize/crop 过程偷偷改变任务定义
* 后续没人知道 target 是怎么来的

## GitHub issue 正文建议

Title:
`Add Stage 4 minimal dataset path and ROI target adapter`

Body:

```md
## Background
Current trusted training code and current optical toy verification code use different target conventions. Stage 4 needs a minimal but real data path that supervises `I_out_roi` using non-synthetic image targets.

## Goal
Build a minimal dataset pipeline and target adapter for Stage 4 closed-loop training.

## Tasks
- Reuse the current trusted EMNIST data path where possible
- Define the Stage 4 HR input format
- Define how to derive the ROI-aligned supervision target
- Save a small visualization preview of input/target pairs
- Keep the protocol minimal and explicitly non-paper-final

## Deliverable
- Stage 4 dataset module
- ROI target adapter
- Preview artifacts for sanity inspection

## Acceptance
- A batch can be loaded successfully
- Target shape matches the optical ROI readout shape
- Saved previews are visually interpretable
```

---

# Issue 4.4 — Implement Stage-4 minimal hybrid model wrapper

## 这个 issue 要解决什么

把 encoder 和 optical decoder 封装成一个明确的 hybrid model。

## 为什么需要 wrapper

虽然 `DiffractiveDecoder` 已经有 `forward_from_phase_provider(...)`，但 Stage 4 还是最好有一个显式 wrapper，把“谁负责什么”写清楚。

## 新文件建议

`src/models/stage4/hybrid_sr_minimal.py`

## 推荐 forward 输出

建议返回一个 dict，至少包含：

* `phi_lr`
* `U0`
* `U_out_full`
* `I_out_full`
* `I_out_roi`

因为 Stage 4 的重心是 debug learnability，中间量不能藏起来。

## 设计要求

* 不修改 optical core
* 只把 encoder 输出接到 `forward_from_phase(...)`
* 支持 `return_intermediates=True`

## 验收标准

* 用 fake input 可以完整跑通
* 输出 dict 中关键中间量齐全
* device / dtype 兼容基本正常

## 风险点

* wrapper 把 contract 搞模糊
* 把 optical 细节偷偷重写
* forward 返回信息太少，后续难 debug

## GitHub issue 正文建议

Title:
`Add Stage 4 hybrid wrapper for encoder-to-optics closed-loop forward`

Body:

```md
## Background
The optical decoder already exposes Stage-4-friendly entrypoints, but Stage 4 still needs a clear hybrid model wrapper for end-to-end training and debugging.

## Goal
Create a minimal hybrid model wrapper that connects:
`HR input -> minimal encoder -> phi_lr -> diffractive decoder`.

## Tasks
- Add a wrapper module under `src/models/stage4/`
- Keep intermediate tensors visible for debugging
- Reuse `forward_from_phase(...)`
- Do not redesign or rewrite the optical core

## Deliverable
- Hybrid Stage 4 model wrapper

## Acceptance
- End-to-end forward works on a fake batch
- Key intermediate tensors are returned
- Optical contract remains unchanged
```

---

# Issue 4.5 — Implement Stage-4 minimal trainer with artifact saving

## 这个 issue 要解决什么

这是 Stage 4 的核心 issue：写出真正的 trainer。

## 直接依据

`PROJECT_CONTEXT.md` 已经明确写了当前 repo 没有 real Stage 4 trainer。

## 复用来源

大量复用 `train_electronic_baseline.py` 的工程外壳，包括：

* CLI
* config merge
* output dir
* checkpoint
* history / summary / metrics
* samples 保存逻辑 

## 新文件建议

`scripts/train_stage4_minimal.py`

## trainer 必须支持的功能

至少要有：

* train / val loop
* optimizer
* checkpoint
* history json
* summary json
* overfit-single-sample 开关
* save-samples 开关

### Stage 4 特有的 artifact

每个 epoch 或每若干 step 保存：

* input HR
* target ROI
* `phi_lr`
* `I_out_roi`
* `|I_out_roi - target|`

如果能多保存一项，建议加：

* encoder grad norm
* decoder grad norm

## loss 建议

首版先用 normalized MAE。因为 Stage 3 toy script 已经用了同类规范化思路，论文主损失本体也是归一化 MAE，只是加了可选 efficiency 项。 

Stage 4 先不引入 efficiency penalty，除非最小闭环完全学不动。

## 验收标准

* 在 CPU 或 CUDA 上能完整跑完一个短训练
* 生成 checkpoint / summary / sample artifacts
* loss 数值可记录
* 中间图可检查

## 风险点

* 一开始把 trainer 写太通用，导致过重
* 偷偷引入 Stage 5 功能
* 不保存中间图，后面无法定位 learnability 问题

## GitHub issue 正文建议

Title:
`Implement Stage 4 minimal trainer for encoder-optics closed-loop training`

Body:

```md
## Background
The repo already has a trusted electronic baseline training skeleton, but it still lacks a real Stage 4 trainer for encoder + optical decoder end-to-end training.

## Goal
Create the first minimal Stage 4 training script for closed-loop learnability validation.

## Tasks
- Add `scripts/train_stage4_minimal.py`
- Reuse the trusted training skeleton style where possible
- Support single-sample overfit mode
- Support small-subset training mode
- Save checkpoints, history, summary, and debug visualizations
- Record gradient statistics for encoder and optical parameters

## Deliverable
- Stage 4 trainer script
- Minimal config if needed
- Saved artifact structure

## Acceptance
- A short run completes successfully
- Artifacts are generated
- Loss/history are recorded
- Intermediate visualizations are readable
```

---

# Issue 4.6 — Add single-sample overfit experiment and acceptance report

## 这个 issue 要解决什么

把 Stage 4 的第一个硬验收做掉：**单样本 overfit**。

## 为什么必须单独成 issue

这是 Stage 4 的第一道门槛。
`plan_overview.md` 已经明说 Stage 4 最重要的问题只有一个：系统有没有梯度驱动下的可学习性；而最小证据之一就是 1 张图能不能过拟合。

## 实验协议建议

* dataset: 1 sample
* train / val 共用同一张
* batch size = 1
* 训练几十到几百 step 即可
* 记录：

  * loss curve
  * target vs output
  * `phi_lr` 演化
  * encoder / decoder grad norm

## 产物

建议目录：

`outputs/stage4/minimal_single_sample_overfit/...`

同时写一个简要报告：

`docs/execution/stage4_single_sample_report.md`

## 验收标准

必须至少满足：

* final loss 明显低于 initial loss
* 输出结构与 target 更接近
* `phi_lr` 不是常数死图
* encoder grad norm 非零
* decoder grad norm 非零

## 风险点

* 光学链能更新，但 encoder 梯度断了
* loss 下降只是靠 scale normalization 投机
* 输出只在能量上变，不在结构上变

## GitHub issue 正文建议

Title:
`Run Stage 4 single-sample overfit sanity and write acceptance report`

Body:

```md
## Background
The first hard acceptance gate of Stage 4 is whether the closed-loop hybrid system can overfit a single sample.

## Goal
Run a single-sample overfit experiment and document whether the system is truly learnable.

## Tasks
- Use the new Stage 4 trainer in overfit-single-sample mode
- Save loss curves and intermediate artifacts
- Check encoder and optical gradient flow
- Write a short execution report

## Deliverable
- Output artifacts for the overfit run
- `docs/execution/stage4_single_sample_report.md`

## Acceptance
- Loss decreases clearly
- Output ROI becomes closer to target
- Encoder gradients and optical gradients are both non-zero
- The report states pass/fail honestly
```

---

# Issue 4.7 — Run small-subset closed-loop training and produce first Stage-4 summary

## 这个 issue 要解决什么

完成 Stage 4 的第二道门槛：**小数据集闭环训练**。

## 为什么这是第二道，不是第一道

因为没有单样本 overfit，就不配上小数据集。
如果 1 张图都学不动，16 张图只会让问题更模糊。

## 实验协议建议

* subset size: 8 / 16 / 32 三选一，建议从 16 起
* batch size: 2 或 4
* epochs: 小规模即可
* 保存同类 artifact

## 需要观察什么

* train loss 是否下降
* val loss 是否至少趋势合理
* 不同样本上输出是否不是同一个模式塌缩
* `phi_lr` 是否对输入有区分性

## 产物

* `outputs/stage4/minimal_small_subset/...`
* `docs/execution/stage4_small_subset_report.md`

## 验收标准

* 小数据集上 loss 明显下降
* 至少多个样本输出存在可辨别差异
* 没有严重模式塌缩
* 报告中明确说明是“Stage 4 可学习性通过/未通过”，不是伪装成论文结果

## 风险点

* 单样本会学，小数据集立刻崩
* 训练看似下降，但全部输出相似
* optical decoder 被 encoder 完全绕开成 trivial 解

## GitHub issue 正文建议

Title:
`Run Stage 4 small-subset closed-loop training and summarize learnability`

Body:

```md
## Background
After single-sample overfit, Stage 4 must verify whether the hybrid system still learns on a small non-trivial subset.

## Goal
Run the first small-subset closed-loop training and summarize whether Stage 4 learnability generalizes beyond a single sample.

## Tasks
- Train on a small subset using the Stage 4 trainer
- Save representative intermediate outputs
- Inspect whether outputs differ across inputs
- Write a concise summary report

## Deliverable
- Small-subset run outputs
- `docs/execution/stage4_small_subset_report.md`

## Acceptance
- Loss decreases on the small subset
- Outputs are input-dependent
- No obvious collapse to a single pattern
- The report gives an honest pass/fail judgment for Stage 4
```

---

# Issue 4.8 — Consolidate Stage-4 docs, configs, and go/no-go decision for Stage 5

## 这个 issue 要解决什么

把 Stage 4 的结果收口，不然前面的实验会散掉。

## 需要同步的文档

至少更新：

* `docs/ai/PROJECT_CONTEXT.md`
* `docs/execution/experiment_log.md`
* `docs/execution/results_summary.md`

如果你已经生成“第四阶段具体规划.md”，这里也要补执行结果回链。

## 要写清楚的内容

最终必须明确回答：

* Stage 4 是否通过
* 如果通过，证据是什么
* 如果没通过，最优先问题是什么
* 是否允许进入 Stage 5
* Stage 5 前还缺什么基础设施

## 验收标准

出现一份明确的 go / no-go 结论，而不是含糊其辞。

## GitHub issue 正文建议

Title:
`Consolidate Stage 4 results and make Stage-5 go/no-go decision`

Body:

```md
## Background
Stage 4 is only meaningful if its outcome is consolidated into the repo's documentation and used to make a clear go/no-go decision for Stage 5.

## Goal
Summarize Stage 4 execution results, sync the docs, and make an explicit Stage 5 entry decision.

## Tasks
- Update PROJECT_CONTEXT with latest Stage 4 status
- Update experiment log and results summary
- Summarize single-sample and small-subset outcomes
- Record known remaining gaps
- Make a clear go/no-go call for Stage 5

## Deliverable
- Updated docs
- Clear Stage 5 gate decision

## Acceptance
The repo documents clearly state whether Stage 4 passed and what evidence supports the decision.
```

---

# 四、建议的推进顺序

最推荐的执行顺序不是按“哪个最想做”，而是按依赖关系走：

## 第 1 步

**Issue 4.1**
先冻结协议，不然后面所有 issue 都会漂。

## 第 2 步

**Issue 4.2 + 4.4**
先把最小 encoder 和 hybrid wrapper 接起来。
这一步结束后，你就应该已经能做 fake input forward。

## 第 3 步

**Issue 4.3**
把真实图像数据接入闭环，哪怕只是最小版本。

## 第 4 步

**Issue 4.5**
写 trainer。
没有 trainer，你现在所有 Stage 4 讨论都只是设计稿。

## 第 5 步

**Issue 4.6**
跑单样本 overfit。
这是第一道实证门槛。

## 第 6 步

**Issue 4.7**
跑小数据集闭环训练。
这是第二道实证门槛。

## 第 7 步

**Issue 4.8**
收口文档，做 go / no-go 决策。

---

# 五、每个 issue 的“工程重点提醒”

为了避免 coder 偏航，我把每个 issue 最容易犯的错也明确指出来。

## Issue 4.1 最容易犯的错

把 Stage 4 protocol 写成“论文设置预案”。

## Issue 4.2 最容易犯的错

一上来硬对齐论文 encoder，导致实现和 debug 成本爆炸。
论文 encoder 结构与学习率量级应记账到 Stage 5。

## Issue 4.3 最容易犯的错

target adapter 写得不透明，后续根本不知道自己在监督什么。

## Issue 4.4 最容易犯的错

wrapper 不返回中间量，导致 learnability 出问题时根本无从定位。

## Issue 4.5 最容易犯的错

trainer 做得太通用，把 Stage 5/6 的需求提前塞进来。

## Issue 4.6 最容易犯的错

只看 loss，不看图。
Stage 4 必须同时看数值与可视化。

## Issue 4.7 最容易犯的错

看到小数据集 loss 降一点就宣布通过，但其实输出全塌缩成同一种模式。

## Issue 4.8 最容易犯的错

结论写得太软，导致项目实际上没有“出关条件”。

---

# 六、建议的 milestone 结构

你可以直接在 GitHub 上这么建：

## Milestone

`Stage 4 — Minimal Closed-Loop Training`

## Issues

* 4.1 Freeze Stage-4 minimal protocol and tensor contract
* 4.2 Implement minimal phase encoder for Stage 4 closed-loop training
* 4.3 Add Stage 4 minimal dataset path and ROI target adapter
* 4.4 Add Stage 4 hybrid wrapper for encoder-to-optics closed-loop forward
* 4.5 Implement Stage 4 minimal trainer for encoder-optics closed-loop training
* 4.6 Run Stage 4 single-sample overfit sanity and write acceptance report
* 4.7 Run Stage 4 small-subset closed-loop training and summarize learnability
* 4.8 Consolidate Stage 4 results and make Stage-5 go/no-go decision

---

# 七、一个更实用的结论

如果我替你拍板，我会这样执行：

先建 **4.1、4.2、4.4、4.5** 四个 issue，作为第一轮开发主线；
**4.3** 可以穿插，但不要拖到最后，因为 trainer 需要真实 target；
等 trainer 跑通后，立刻做 **4.6 单样本 overfit**；
只有 4.6 通过，才进入 **4.7 小数据集闭环**；
最后 **4.8** 收口。

也就是说，Stage 4 的真实主线其实是：

> **协议冻结 → encoder 接入 → hybrid wrapper → trainer → 单样本过拟合 → 小数据集闭环 → 文档收口**

这条线是当前 repo 最合逻辑、最省试错成本的推进方式。

下一步最值得做的，就是我直接给你把这 8 个 issue 全部写成可粘贴到 GitHub 的正式 markdown 模板。
