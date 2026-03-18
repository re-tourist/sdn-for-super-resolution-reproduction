你正在当前仓库中实现已有任务：

[Task3-7] feat: add clean phase-provider hook for future Stage 4 integration

请先明确理解本任务边界：
- Stage 4 未来会需要 encoder
- 但 Stage 3 当前不能提前实现 encoder / joint trainer / end-to-end training
- 本任务只做“phase-provider hook + 接口约束 + 最小 stub/adapter”
- 目标是让 optical core 与未来 encoder 的实现解耦
- 绝不能借此任务偷偷进入 Stage 4

你必须严格遵循：
- AGENTS.md
- docs/plan/stage3_contract_freeze.md
- docs/integration/stage3_unresolved_params.md
- docs/plan/stage3_sdn_porting_plan.md
- 当前仓库中 Stage 3 已实现的 optics core / readout / validation 代码
- 当前 Task3-7 issue 的描述：
  - 定义 forward_from_phase
  - 定义 forward_from_field
  - 写明 encoder hook 约束
  - deliverables 是接口说明 + 最小 stub/adapter
  - acceptance criteria 是 optical core 与 encoder 实现解耦

当前已知前提：
- 当前 optical decoder 已具备稳定的 Stage 3 contract
- 现有主链路已经落到：
  phase -> field -> optical propagation -> intensity -> ROI
- Stage 3 当前验证逻辑已完成，不要回头重写 Issue 2~6
- 本任务不是验证任务，而是接口整理任务

本次任务目标：
1. 明确 optical module 的两类输入入口：
   - forward_from_phase(...)
   - forward_from_field(...)
2. 增加一个 clean phase-provider hook / adapter / protocol
3. 写清楚 future encoder 若要接入，需要满足什么 contract
4. 保持当前 Stage 3 行为不变，不破坏现有实验脚本
5. 不引入 Stage 4 训练系统

你要完成的核心工作：

A. 稳定 forward_from_phase(...) 的语义
要求：
1. 输入是 phase-domain tensor（例如 phi_lr 或其等价 phase representation）
2. optical module 内部负责 phase -> field 的映射
3. 然后进入已有 propagation/readout chain
4. 接口注释中必须写清楚：
   - 输入 shape 语义
   - dtype/device 约束
   - batch 维约定
   - phase mapping 的责任归属

B. 稳定 forward_from_field(...) 的语义
要求：
1. 输入是已经构造好的 coherent complex field
2. 跳过 phase mapping
3. 直接进入 propagation/readout chain
4. 接口注释中必须写清楚：
   - field 必须是 complex tensor
   - shape 语义
   - 与 forward_from_phase 的职责区别

C. 增加最小 phase-provider hook / adapter / protocol
要求：
1. 只是一个“谁来提供 phase”的接口约定，不是真正 encoder 实现
2. 可以采用最轻量方案，例如：
   - typing.Protocol
   - callable contract
   - 小型 adapter/stub class
3. 它应该表达：
   - 给定某种上游输入，provider 产出 phase tensor
   - optical decoder 只消费 phase，不关心 phase 是手工提供还是将来由 encoder 生成
4. 禁止把训练逻辑、损失逻辑、dataset 逻辑塞进去

D. 写清楚 contract 文档 / 注释
至少应明确：
1. phase provider 输出 tensor 的语义
2. 期望 shape / dtype / device
3. 是否允许 batch 维
4. optical module 对 phase range / mapping 的假设
5. 当前 Stage 3 默认仍是“外部直接提供 phase”
6. future Stage 4 可以在什么位置接 learned encoder，但当前不实现它

推荐改动范围（尽量小）：
- src/models/optics/diffractive_decoder.py
- 如有必要，新增一个非常小的接口文件，例如：
  - src/models/optics/phase_provider.py
  或
  - src/models/interfaces/phase_provider.py
- 如有必要，补少量开发文档或 README 注释
不要做大规模目录重构。
不要重写 propagation.py / readout.py 的核心数值逻辑。

必须遵守的边界：
1. 不实现真正 encoder
2. 不实现 joint model
3. 不实现 end-to-end trainer
4. 不新增训练循环
5. 不修改 Stage 3 的 optical/readout/crop contract
6. 不破坏已有 Issue 2~6 的实验行为
7. 不做过度抽象（例如复杂 registry/factory/plugin framework）
8. 不因为“为未来准备”而重构整个 models 目录

实现风格要求：
- small, local, readable changes
- clear naming
- 接口注释优先，把语义写清楚
- 明确标注：
  - 哪些属于当前 Stage 3 稳定 contract
  - 哪些只是 future Stage 4 integration hook
- 尽量保证现有脚本无需修改或仅最小修改即可继续运行

建议最小验收检查：
1. 现有 forward_from_phase(...) 仍可正常工作
2. 现有 forward_from_field(...) 仍可正常工作
3. 新增 provider hook / adapter 不影响现有 Stage 3 sweep / fitting 脚本
4. 从代码结构上能清楚看出：
   - optical core 不负责生成 phase
   - future encoder 只需满足 provider contract 即可接入
5. 没有引入任何 joint training / Stage 4 trainer 代码

本次交付物：
- 稳定后的 forward_from_phase / forward_from_field 接口
- 一个最小的 phase-provider hook / adapter / protocol
- 必要的接口说明 / 注释 / 小文档更新
- 如环境允许，可补一个极小的 smoke-level usage example，但不要扩展成新训练脚本

输出要求：
1. 先简短复述你理解到的任务边界
2. 列出你准备新增/修改的文件
3. 再实施代码修改
4. 最后给出：
   - 变更总结
   - phase-provider hook 的接口说明
   - 当前刻意未实现的 Stage 4 内容
   - 风险点与后续建议