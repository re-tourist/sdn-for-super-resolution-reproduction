你正在当前仓库中实现：

Issue 4.3 — Build Stage-4 minimal dataset path and target adapter

当前状态：
- Stage 3 optical contract 已冻结：
  phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi
- minimal phase encoder 已完成
- Stage 4 minimal hybrid wrapper 已完成，并可在 fake batch 上跑通：
  x_hr -> encoder -> phi_lr -> decoder.forward_from_phase(...) -> I_out_roi
- 当前缺口不是 optical core，也不是 wrapper，而是：
  1) 最小真实数据路径
  2) ROI-aligned supervision target adapter

你必须严格遵循：
- docs/ai/PROJECT_CONTEXT.md
- docs/plan/plan_overview.md
- docs/plan/stage3_contract_freeze.md
- docs/paper/paper_notes.md
- 当前仓库内已落地的:
  - Stage 3 optical module
  - minimal phase encoder
  - minimal hybrid wrapper
  - trusted EMNIST path / baseline path
- 不重新设计 optical core
- 不提前做 trainer / loss loop / Stage 5 paper alignment
- 保持 protocol 最小、真实、可调试，并明确 non-paper-final

本次任务目标：
构建 Stage 4 的最小真实数据路径，并定义 target adapter，
让后续 closed-loop training 能监督 wrapper 输出的 I_out_roi。

本次需要回答的核心问题：
1. Stage 4 的 HR input tensor 到底长什么样？
2. target_roi 如何从真实图像目标中得到？
3. target 的 shape 和语义如何与 optical I_out_roi 对齐？
4. 能否保存少量 preview，确认 input/target 配对是可解释的？

你要完成的核心工作：

A. 复用可信 EMNIST 数据路径
要求：
1. 尽量复用当前 trusted EMNIST path，而不是重新发明 dataset
2. 只做最小 Stage 4 数据壳
3. 明确 Stage 4 当前使用的输入图像语义：
   - x_hr 是什么
   - target_roi 是什么
4. 不把数据模块写成 paper-final pipeline

B. 定义 Stage 4 HR input format
要求：
1. 明确 x_hr 的 shape 约定，例如 [B, 1, H_hr, W_hr]
2. 明确其数值范围 / dtype / device 约束
3. 明确是否需要 resize / pad / normalize
4. 写清当前协议只是 Stage 4 minimal learnability check，不是论文最终 HR protocol

C. 实现 ROI target adapter
要求：
1. 给定真实 target 图像，导出可与 I_out_roi 对齐的 supervision target
2. target shape 必须匹配 optical ROI readout shape
3. crop / resize / normalization 的逻辑必须显式，不允许隐藏在脚本散代码里
4. 命名清楚，建议有单独 adapter/helper，而不是把切片逻辑散落在训练脚本中
5. 要和 Stage 3 已冻结的 ROI contract 一致：
   - I_out_full 保留用于诊断
   - 主要监督对象是 I_out_roi

D. 保存 preview artifacts
至少保存：
1. input x_hr preview
2. target_roi preview
3. 如方便，可保存 input/target pair preview grid
目标是人工能快速判断：
- 数据是不是读对了
- target adapter 有没有明显错位
- shape/语义是否可解释

推荐落地文件（按需最小化）：
- src/datasets/stage4_emnist.py
- src/datasets/target_adapter.py
- 或等价的小型模块化落点
- 如确有必要，可加一个很小的 preview script
不要大规模重构 repo。

必须遵守的边界：
1. 不实现 trainer
2. 不实现完整 loss loop
3. 不改 optical core
4. 不改 minimal hybrid wrapper 的核心职责
5. 不把 4.3 做成 paper-final data pipeline
6. 不把 target adapter 写成隐式、分散、难追踪的脚本逻辑
7. 不编造 preview 或 batch 结果；跑不通就诚实说明

建议最小验收检查：
1. 一个 batch 能加载成功
2. x_hr shape 符合 wrapper 预期
3. target_roi shape 与 I_out_roi shape 一致
4. saved previews visually interpretable
5. target adapter 逻辑可被后续训练脚本直接复用

实现风格要求：
- small, local, readable changes
- clear naming
- 把“真实最小数据路径”和“论文最终数据协议”明确区分
- 注释里写清楚当前 target adapter 的责任边界

输出要求：
1. 先简短复述你理解到的 4.3 边界
2. 列出准备新增/修改的文件
3. 再实施代码修改
4. 最后给出：
   - 变更总结
   - Stage 4 x_hr / target_roi 的当前定义
   - preview 保存位置与内容
   - batch / shape 自检结果
   - 下一步建议（应指向最小训练脚本，而不是回头再改 wrapper）