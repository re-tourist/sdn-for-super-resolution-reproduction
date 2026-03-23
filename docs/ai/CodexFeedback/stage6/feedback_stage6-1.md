已新增 Stage 6 的 debug protocol freeze 文档，冻结了 Stage 6 的 debugging-first 边界，并把失败分类、证据阶梯、issue 权限、artifact 协议和退出条件写清楚，供 `6.2` 到 `6.8` 直接继承。

精确变更文件：
- [stage6_debug_protocol.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/docs/plan/stage_plan/stage6/stage6_debug_protocol.md)

新 debug protocol 文档之外的 consistency edit：
- 无。我没有修改 `stage6_plan.md` 或 `stage6_issue_plan.md`。

最重要的冻结决策：
- Stage 6 现在只允许围绕现有 Stage 5 paper-path 基线做定向排错；更宽的 systematic experiments 仍然被冻结，直到 `Issue 6.8` 明确重新批准。
- `Issue T003` 被冻结为“已确认存在 scale ambiguity 机制”，但不是“唯一根因已证实”。
- 每个 Stage 6 run 都必须显式区分问题 A：`raw intensity collapse`，以及问题 B：`normalized reconstruction failure`。
- raw preview 必须保留为第一类工件；sigma-rescaled 或 normalized 视图只能作为诊断补充，不能替代 raw 视图。
- Issue 权限已经冻结：`6.2` / `6.3` 只做语义审计和观测增强，`6.4` / `6.6` 只做受控复现，`6.5` 才能改 loss-like 的 scale handling，`6.7` 才能在门槛满足后重开 paper-sensitive geometry。

最重要的、刻意保留未决的假设：
- 修复 scale-related collapse 之后，`normalized reconstruction failure` 是否仍然存在。
- 当前主失败是否主要是 `L=5` 特有，还是 paper-path 更普遍的问题。
- `distance mapping`、`phase range`、`crop / FOV alignment` 等 paper-sensitive 假设是否需要在后续证据充分后重审。

这是文档任务，未运行测试。
