# Stage 6 详细工程计划

## 1. 文档定位

本文档定义本仓库在 Stage 5 收口之后进入 `Stage 6 — 系统化实验、排错与总结` 的工程推进方案。

它的目标不是重新设计 Stage 5，而是回答下面两个问题：

1. 在已经冻结的 paper-aligned pipeline 之上，应该先做哪些系统化实验。
2. 如何把实验矩阵控制在可解释、可复盘、可收口的范围内，而不是重新掉回“边写边试”的状态。

本文档优先级低于：

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/paper/paper_notes.md`
4. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
5. `docs/execution/stage5_smoke_report.md`
6. `docs/execution/stage5_main_run_report.md`

若 Stage 5 收口文档与本计划冲突，以 Stage 5 收口结论为准；Stage 6 不得反向篡改 Stage 5 已冻结的主线协议。

---

## 2. 当前上下文与 Stage 6 触发条件

截至当前仓库状态，已知事实是：

- Stage 3 optical contract 已冻结：`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- Stage 4 learnability gate 已通过
- Stage 5 数据、optics config、encoder、loss、trainer、regular eval、blind eval、smoke run 路径已经落地
- Stage 5 主线 `L=5 phase-only` main-run 路径已准备为可在 Linux server 上执行

但 Stage 6 不能默认“立即全量开始”。进入 Stage 6 前，先区分三个入口级别：

### 2.1 Entry Level E1: engineering-ready

满足条件：

- Stage 5 pipeline 可启动
- smoke run 已通过
- Stage 5 文档已把默认值、未决项和 artifact 路径写清楚

允许开始：

- Stage 6 文档规划
- 轻量级结果注册表搭建
- 不依赖新训练的 eval-only 预备工作

不允许开始：

- 把跨 depth / 跨 modulation / 跨 quantization 的大矩阵训练直接跑起来

### 2.2 Entry Level E2: baseline-evidenced

满足条件：

- 至少一次真实的 `L=5 phase-only` Stage 5 main run 已在 Linux server 上执行
- `val / test / blind eval` 至少形成一套可复盘记录
- `docs/execution/stage5_main_run_report.md` 已写入真实结果或真实中断点

允许开始：

- 依赖已训练 checkpoint 的 Stage 6 eval-only 工作
- 以 `L=5 phase-only` 为锚点的 Stage 6 P0 实验

### 2.3 Entry Level E3: baseline-settled

满足条件：

- Stage 5.11 已完成
- Stage 5 主线结果与资源边界已经被清楚区分
- 仓库已明确 Stage 6 `GO`

允许开始：

- 全量 Stage 6 训练型对照和系统化 ablation

结论：

- 没有 E2 证据时，不应把 Stage 6 写成真正的执行阶段
- 没有 E3 决策时，不应启动大规模训练矩阵

---

## 3. Stage 6 核心目标

Stage 6 的主任务不是“继续把 Stage 5 做完”，而是围绕论文主张建立系统化证据。

本阶段要回答的核心问题是：

1. `L=1/3/5` 的 paper-aligned optical depth 是否在统一协议下呈现论文声称的趋势。
2. efficiency term 的开关是否对结果有稳定、可解释的影响。
3. phase-only 与 complex-valued 的差距在本复现系统里是否成立。
4. 量化对 blind 高频恢复的影响是否复现论文趋势。
5. 系统对错位 / 失配是否敏感，敏感性主要来自哪里。
6. 如果结果与论文不一致，问题更可能来自数据频谱、训练预算、读出定义还是物理实现细节。

---

## 4. Stage 6 非目标

以下内容不应在 Stage 6 开始时一次性塞满：

- 大而全的超参搜索
- 重新设计 Stage 5 trainer / eval framework
- 修改 Stage 5 的 crop、normalization、dataset split 作为“隐含修复”
- 没有主线基线时就并发跑一整套训练矩阵
- 把 THz 实验硬件链路复现与数值系统化实验混成一个阶段

---

## 5. 规划原则

### 5.1 先冻结比较基线，再做变量控制

Stage 6 每一类实验都必须有一个明确的 anchor baseline：

- 默认 anchor：Stage 5 `L=5 phase-only` main run
- 除非某个 issue 明确声明要改变量，否则：
  - dataset protocol 不变
  - eval splits 不变
  - crop / FOV semantics 不变
  - regular eval + blind eval 口径不变

### 5.2 一次只改一个主要因素

推荐控制方式：

- depth study 只改 `L`
- efficiency study 只改 efficiency-term 开关或 `gamma` policy
- modulation study 只改 encoder/modulation representation
- quantization study 只改 quantization bit-width
- misalignment study 只改 displacement perturbation

若一次改多个因素，必须单独标注为“组合实验”，不能混入主矩阵。

### 5.3 先 eval-only，再 training-heavy

能通过冻结 checkpoint 回答的问题，不先上重新训练。

因此推荐顺序是：

1. baseline 结果登记
2. eval-only sweep
3. 小规模训练型 ablation
4. 成本最高的 full main-run 对照

### 5.4 Stage 6 新变体不得污染 Stage 5 主线

要求：

- Stage 5 configs 保持只读
- Stage 6 变体配置放在 `configs/stage6/`
- Stage 6 输出放在 `outputs/stage6/`
- Stage 6 报告单独写入 `docs/execution/` 或 `docs/plan/stage_plan/stage6/`

---

## 6. Stage 6 工作流总览

按优先级，Stage 6 建议拆成 `P0 / P1 / P2` 三层。

### 6.1 P0: 必做主线

P0 直接对应论文主张和当前 repo 最需要补齐的系统化证据：

1. baseline freeze 与 experiment registry
2. `L=1/3/5` depth comparison
3. efficiency on/off ablation

### 6.2 P1: 高价值扩展

P1 直接来源于论文或补充材料，但可以在 P0 之后排队：

1. phase-only vs complex-valued
2. optional amplitude-only supplemental check
3. quantization inference sweep

### 6.3 P2: 延后项

P2 要么更贵，要么更偏 robustness / hardware-aware：

1. misalignment robustness evaluation
2. optional misalignment vaccination training
3. targeted root-cause reruns for unresolved failures

---

## 7. 详细执行顺序

### 7.1 Stage 6A — Freeze experiment registry and artifact protocol

目标：

- 在开始系统化实验前，先把 Stage 6 的结果登记方式固定下来

任务：

1. 定义 Stage 6 run family 命名规则
2. 规定每次实验必须保存的最小工件
3. 定义聚合表字段
4. 明确哪些实验依赖训练、哪些只依赖已有 checkpoint

建议产物：

- `docs/plan/stage_plan/stage6/stage6_plan.md`
- `docs/plan/stage_plan/stage6/stage6_experiment_registry.md`
- `outputs/stage6/summary/stage6_matrix.csv` 或等价聚合文件

最小工件要求：

- `config_snapshot.json`
- `run_summary.json` 或 `summary.json`
- 关键 metrics
- blind eval summary
- 指向源 checkpoint 的路径或副本
- 运行命令
- 如涉及训练，还要有 `history`、`grad_stats`、checkpoint 信息

验收：

- 后续 Stage 6 issue 不需要各自重新定义 artifact 协议

### 7.2 Stage 6B — Run paper-aligned depth comparison for `L=1/3/5`

目标：

- 在统一 Stage 5 主线协议下验证 `L=1/3/5` 的深度趋势

固定项：

- dataset protocol
- encoder family
- loss definition
- eval protocol
- blind line-pair target family

变量：

- optics depth `L`
- 对应的 paper-aligned distance schedule

执行策略：

1. 先确定 `L=1/3/5` 各自 main config
2. 至少跑到“可比较的 stop point”
3. 对每个 depth 执行 val / test / blind eval
4. 先做单 seed 比较
5. 只有当结论接近或不稳定时，才增加第二个 seed

要回答的问题：

- 是否出现 `L=5 > L=3 > L=1` 的总体趋势
- 差异主要体现在 PSNR / SSIM，还是 blind 高频恢复
- 是否存在 depth 越深但训练更不稳的工程代价

验收：

- 同一报告中能横向比较 `L=1/3/5`
- depth 结论与资源约束被明确分开

### 7.3 Stage 6C — Efficiency-term on/off ablation

目标：

- 验证 efficiency penalty 是否显著影响输出质量或能量集中

固定项：

- 先固定一个主 depth，默认 `L=5`
- 若资源允许，再扩展到 `L=1` / `L=3`

变量：

- `efficiency_term.enabled`
- 必要时记录 `gamma_by_depth`

执行策略：

1. 先以 Stage 5 baseline 为 anchor
2. 在相同训练预算下比较 on/off
3. regular metrics 与 blind eval 一起记录
4. 如果 quality 与 efficiency 出现 trade-off，单独写明，不混成“谁更好”的一句话

要回答的问题：

- efficiency term 是帮助了 ROI 内成像，还是主要改变了能量分布
- 它改善的是 main metrics、blind 高频恢复，还是仅改善物理可解释性

验收：

- 有一份明确对比 on/off 的结果表
- `gamma` 与 switch 状态均可追溯

### 7.4 Stage 6D — Modulation-mode comparison

目标：

- 检查 paper notes 里“complex-valued 略优于 phase-only”的趋势是否在当前 repo 中成立

建议优先级：

1. phase-only vs complex-valued
2. amplitude-only 只作为补充项，不提前升格为主线

约束：

- 不得借该 issue 重写 optical contract
- 如果 complex-valued 路径要求大改 Stage 3 core，应先停下补 design note

执行策略：

1. 先实现最小 modulator representation 差异
2. 用同一 depth、同一 dataset、同一 eval path 比较
3. phase-only 继续作为 reference baseline
4. amplitude-only 只在 complex-valued 路径可控后再补

要回答的问题：

- complex-valued 的收益是否稳定
- 收益主要体现在哪类指标
- 更高自由度带来的收益是否值得其实现复杂度

验收：

- 至少有 phase-only vs complex-valued 的直接对照
- amplitude-only 若未做，也要在文档中明确记为延后项，而不是静默缺失

### 7.5 Stage 6E — Quantization sweep

目标：

- 复现论文对 phase quantization 的定性趋势

依据：

- `docs/paper/paper_notes.md` 建议第一版先做 inference-time quantization

推荐顺序：

1. 用连续 / 高 bit checkpoint 作为源
2. 在推理阶段测试 `16 / 8 / 6 / 4 / 2 bit`
3. regular eval 只作为补充
4. blind 高频结果作为主观察对象

明确不默认做：

- quantization-aware retraining

只有当以下条件满足时才考虑追加：

- inference sweep 已完成
- 4-bit / 2-bit 明显退化
- 用户明确希望继续追 QAT 改善

要回答的问题：

- 8-bit / 6-bit 是否基本可接受
- 4-bit / 2-bit 是否出现论文描述的明显退化
- 退化首先体现在 blind 高频，还是常规 PSNR / SSIM

验收：

- bit-width 趋势清楚
- 每个 bit-width 的 quantization policy 被显式记录

### 7.6 Stage 6F — Misalignment robustness evaluation

目标：

- 在不引入真实硬件链路的前提下，先做数值错位鲁棒性检查

依据：

- `paper_notes.md` 已记录 `Δx / Δy / Δz` 的参考量级

推荐顺序：

1. 先做 inference-time perturbation test
2. 再决定是否需要 vaccination training

变量：

- lateral shift
- axial shift
- perturbation sampling distribution

不应默认做：

- 一上来就训练 misalignment-vaccinated 主模型

要回答的问题：

- 当前最佳 baseline 对错位有多敏感
- 敏感性来自哪一层面：depth、quantization、modulation、blind target 类型

验收：

- 至少有一份 robustness 曲线或表格
- training-free robustness 与 training-aware robustness 被清楚区分

### 7.7 Stage 6G — Root-cause troubleshooting and targeted reruns

目标：

- 把“实验结果不符合论文”从现象层推进到原因层

排查优先级建议：

1. 数据频谱覆盖不足
2. 主线训练预算不够
3. crop / normalization / readout 口径不一致
4. blind target 生成方式与论文不一致
5. physics config 偏差

工作方式：

- 针对失败模式开窄 issue
- 只做有假设支撑的 targeted rerun
- 结论进入 `docs/execution/troubleshooting.md`

验收：

- 每个 rerun 都绑定明确假设
- 不再出现“为了看看会不会好”式的无边界试验

### 7.8 Stage 6H — Consolidation and stage summary

目标：

- 把 Stage 6 的系统化结果沉淀为可交付结论

必须更新：

- `docs/execution/experiment_log.md`
- `docs/execution/results_summary.md`
- `docs/execution/troubleshooting.md`
- 相关 stage6 报告

建议补充：

- 一个 stage6 汇总表
- 一个“论文结论复现状态”对照表

验收：

- 能清楚回答“哪些论文趋势复现了，哪些没有，原因最可能是什么”

---

## 8. Stage 6 推荐 issue 家族

下面的 issue 家族不是强制一字不改，但建议作为 Stage 6 拆分起点：

1. `Issue 6.1 — Freeze Stage 6 experiment registry and artifact protocol`
2. `Issue 6.2 — Run paper-aligned depth comparison for L=1/3/5`
3. `Issue 6.3 — Run efficiency-term on/off ablation on the frozen Stage 5 baseline`
4. `Issue 6.4 — Compare phase-only and complex-valued modulation paths`
5. `Issue 6.5 — Run inference-time phase quantization sweep with blind-eval reporting`
6. `Issue 6.6 — Evaluate misalignment robustness on frozen baseline checkpoints`
7. `Issue 6.7 — Summarize Stage 6 findings and update troubleshooting/results docs`

如果资源不足，推荐优先顺序是：

1. `6.1`
2. `6.2`
3. `6.3`
4. `6.5`
5. `6.4`
6. `6.6`
7. `6.7`

这样排序的原因是：

- `L=1/3/5` 与 efficiency ablation 直接对应论文主张
- quantization 可优先用已有 checkpoint 做推理测试
- complex/amplitude-only 需要额外实现成本
- misalignment 更偏后期 robustness

---

## 9. 目录与工件约定

建议 Stage 6 期间使用以下约定：

### 9.1 Configs

- `configs/stage6/`

用途：

- 存放所有 Stage 6 变体 config
- 不覆盖 Stage 5 主线 config

### 9.2 Outputs

- `outputs/stage6/<family>/<run_name>/`

其中 `<family>` 建议使用：

- `depth_compare`
- `efficiency_ablation`
- `modulation_compare`
- `quantization_eval`
- `misalignment_eval`
- `summary`

### 9.3 Docs

- `docs/plan/stage_plan/stage6/`
- `docs/execution/`

原则：

- 计划与 issue 拆分放在 `docs/plan/stage_plan/stage6/`
- 真实执行结果与报告放在 `docs/execution/`

---

## 10. 风险与资源边界

### 10.1 主要风险

1. 把 Stage 6 写成无限扩张的实验池
2. 还没固定 baseline 就开始横向比较
3. 训练型实验和 eval-only 实验混在一起，导致预算失控
4. 结果表没有统一字段，后面无法做归纳
5. 因为论文含糊，就在不同实验里偷偷改变 protocol

### 10.2 资源控制规则

1. 每个训练型 family 先跑单 seed
2. 只有在排序接近或不稳定时才加 seed
3. 优先跑能直接复用 checkpoint 的 eval-only family
4. 对每个 family 先设定 stop point，再谈是否扩预算
5. 任何未跑满预算的 run 都必须记录 exact stop point

---

## 11. Stage 6 完成标准

Stage 6 最低完成标准：

1. 至少一个 Stage 6 结果注册表已建立
2. `L=1/3/5` depth comparison 已完成到可比较程度
3. efficiency on/off ablation 已完成至少一组稳定对照
4. quantization inference sweep 已完成
5. 至少一份 troubleshooting 文档把失败模式和原因假设写清楚
6. 结果摘要能够明确区分：
   - 论文趋势已复现
   - 论文趋势部分复现
   - 尚未复现且原因未定

更强完成标准：

1. phase-only vs complex-valued 已完成
2. amplitude-only 至少有补充说明
3. misalignment robustness 已形成曲线或表格
4. Stage 6 已能支撑最终复现结论整理

---

## 12. 一句话结论

Stage 6 的正确打开方式不是“继续往系统里堆功能”，而是：

先用 Stage 5 冻结好的 paper-aligned baseline 作为锚点，再按 `depth -> efficiency -> quantization -> modulation -> robustness` 的顺序，逐类建立可比较、可解释、可收口的证据链。
