已只更新文档，没有修改任何源码或实现代码。

修改的文件：
- [AGENTS.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/AGENTS.md#L35)
- [docs/integration/stage3_unresolved_params.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/docs/integration/stage3_unresolved_params.md#L30)
- [docs/plan/stage3_contract_freeze.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/docs/plan/stage3_contract_freeze.md#L78)

更新的内容：
- [docs/integration/stage3_unresolved_params.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/docs/integration/stage3_unresolved_params.md#L30) 新增了 `Finalized Stage-3 Decisions`，把 5 个 Immediate Decisions 全部标记为 `RESOLVED`，并移出“待拍板”语境；同时保留 `P1` / `P2` 作为仍可后续收敛的事项。
- [docs/integration/stage3_unresolved_params.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/docs/integration/stage3_unresolved_params.md#L63) 更新了使用说明，明确先执行 finalized decisions，再处理剩余 P1/P2。
- [docs/plan/stage3_contract_freeze.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/docs/plan/stage3_contract_freeze.md#L86) 在 contract 部分明确写入了 `L` 语义、distance schedule schema、center-crop ROI contract、ROI-only supervision 规则，以及 normalized MAE 的 Stage-3 默认实现。
- [docs/plan/stage3_contract_freeze.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/docs/plan/stage3_contract_freeze.md#L109) 新增了显式 `distance_schedule` schema。
- [AGENTS.md](D:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/AGENTS.md#L35) 在 Stage-3 planning freeze 小节下加入了短说明，要求后续 agent 先遵循 `stage3_contract_freeze.md` 和 `stage3_unresolved_params.md`。

确认：
- 5 项人审通过的 Stage-3 决策现在都已经记录为 finalized。
- 文档中已明确这 5 项不再属于待拍板项。
- 本次没有扩展 Stage-3 scope，也没有引入新架构。