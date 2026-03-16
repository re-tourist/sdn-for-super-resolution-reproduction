你正在当前仓库中继续推进已有任务：

[Task3-8] Stage3 模块验证文档与设计记录更新

当前前提：
- Stage 3 contract freeze 已完成
- optical propagation core / readout / crop / validation 链路已完成
- decoder-only single-sample 与 small-subset sweep 已完成
- Task3-7 phase-provider hook 已完成
- 当前任务不是继续写模型代码，而是把 Stage 3 的关键实现决策、验证结果、设计边界和已知问题沉淀到文档中

你必须严格遵循：
- AGENTS.md
- docs/plan/stage3_contract_freeze.md
- docs/integration/stage3_unresolved_params.md
- docs/plan/stage3_sdn_porting_plan.md
- 当前仓库内已经落地的 Stage 3 代码与脚本
- 当前已有实验结果与 summary 文件
- 不要重新设计系统
- 不要改 Stage 3 contract
- 不要伪造未运行的实验结果

本次任务目标：
把 Stage 3 的关键决定固定下来，并更新以下文档：
- docs/execution/experiment_log.md
- docs/execution/results_summary.md
- docs/execution/design_notes.md

至少要记录以下内容：
1. propagation 实现采用什么模型/primitive
2. grid / padding / readout / crop 如何定义
3. phase 参数化方式
4. 单层/多层 decoder 的最小测试结果
5. 已知数值问题 / 当前限制
6. phase-provider hook 的接口边界（只需简要记载，不要展开成 Stage 4 设计文档）

你要完成的核心工作：

A. 更新 docs/execution/experiment_log.md
要求：
1. 记录 Stage 3 的关键执行路径和里程碑
2. 至少包括：
   - Issue 2 optical propagation core
   - Issue 3 readout/crop contract
   - Issue 4 forward sanity
   - Issue 5 decoder-only single-sample fitting
   - Issue 6 decoder-only small-subset depth sweep
   - Task3-7 phase-provider hook
3. 对每项记录：
   - 做了什么
   - 为什么做
   - 得到什么最小结论
4. 语言应简洁、可追溯，避免空话

B. 更新 docs/execution/results_summary.md
要求：
1. 汇总 Stage 3 当前已获得的“事实性结果”
2. 至少包含：
   - L=1/3/5 depth sweep 的 fixed-protocol 结果摘要
   - success_count / total_count / mean_best_loss / mean_loss_decrease 等关键指标
   - 对结果的克制性解释：
     - 可以说明“当前 tiny protocol 下三种 depth 都具备可优化容量”
     - 不能夸大为“论文性能已复现”或“depth 优势已严格证明”
3. 明确哪些结论是当前可说的，哪些结论仍不能说

C. 更新 docs/execution/design_notes.md
要求：
1. 固化 Stage 3 的关键设计决策
2. 至少包括：
   - propagation primitive 的来源与当前实现方式
   - L 的语义：trainable diffractive phase masks 数量
   - distance schedule 的显式语义
   - full-grid forward + ROI supervision contract
   - intensity readout / center crop 的定义
   - phase -> field 的责任归属
   - forward_from_phase / forward_from_field / forward_from_phase_provider 的职责边界
   - phase-provider contract 的最小定义
3. 明确哪些是 Stage 3 stable contract，哪些只是 future Stage 4 integration hook
4. 不要把 design notes 写成愿景文档，重点记录“当前仓库已实现并确认的边界”

D. 文档写作要求
1. 只记录真实存在的实现与实验
2. 若某项只是最小 smoke / tiny protocol 验证，要明确写清，不要夸大
3. 文档结构清晰，最好使用小标题和简短列表
4. 允许补充少量 cross-reference，但不要大改整个 docs 结构
5. 不要顺手重写无关文档

必须遵守的边界：
1. 本次以文档更新为主，不做新的模型功能开发
2. 不新增 Stage 4 训练系统
3. 不修改 Stage 3 物理实现，除非为了修正文档与代码不一致的明显小问题
4. 不编造未跑过的实验
5. 不把 Task3-8 变成“大规模 README 重构”

建议输出风格：
- experiment_log.md：按 issue/task 时间线记录
- results_summary.md：按“当前已验证事实 / 当前不能得出的结论 / 已知限制”组织
- design_notes.md：按“数据流 / 光学实现 / readout contract / provider hook / known issues”组织

建议最小验收标准：
1. 三个文档都被更新
2. 文档内容与当前代码/实验结果一致
3. 能让新对话或新协作者快速理解：
   - Stage 3 做了什么
   - 现在哪些点已经固定
   - 还有哪些数值风险和结论边界
4. 不夸大验证结果
5. Task3-7 的 hook 只被记录为接口预留，而不是已进入 Stage 4

输出要求：
1. 先简短复述你理解到的 Task3-8 边界
2. 列出准备修改的文档文件
3. 再实施文档更新
4. 最后给出：
   - 各文档新增/更新了哪些部分
   - 哪些 Stage 3 结论已被固定记录
   - 哪些内容被刻意留到 Stage 4