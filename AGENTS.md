# AGENTS.md

## 项目定位
本项目用于复现一篇基于衍射神经网络（Diffractive Neural Network, D2NN）的图像超分辨率论文。  
项目目标不仅是“跑出结果”，更重要的是建立一套可复现、可排错、可扩展的实验工程系统。

本项目具有以下特征：
- 研究型/实验型项目，而非普通应用软件
- 包含数据协议、评估协议、模型实现、实验记录等多个层面
- 强调阶段式推进，不鼓励跳步骤堆复杂实现
- 使用 AI coding agent（如 Codex）协助开发，因此需要额外的协作约束

---

## 当前阶段
当前阶段请以 `docs/reproduction_plan.md` 中定义的阶段为准。

如果未特别说明，默认优先顺序如下：
1. 阶段0：项目初始化与工程骨架搭建
2. 阶段1：建立 baseline
3. 阶段2：验证数据与评估协议
4. 阶段3：实现并验证光学模块
5. 阶段4：最小闭环训练
6. 阶段5：论文设定对齐
7. 阶段6：系统化实验、排错与总结

### 阶段约束
在阶段0~2期间：
- 不要提前实现复杂光学模块
- 不要跳过 baseline
- 不要提前做大规模调参
- 不要在论文未明确的地方默认加入复杂 trick
- 优先保证数据管线、评估协议、训练骨架的清晰与可验证

### Stage 3 planning freeze 关键文档
Immediate design decisions have been finalized. Future agents must follow:
- `docs/plan/stage3_contract_freeze.md`
- `docs/integration/stage3_unresolved_params.md`
当任务进入 Stage 3（Optical Module Verification）的前置整理、contract freeze 或 upstream porting 评估时，下列文档已成为主要决策输入。开始实现前，应优先对齐这些文档，而不是先写光学代码：

- `docs/integration/sdn_inventory.md`：盘点 `external/sdn_upstream` 中哪些 optics 能力可参考、哪些任务模块必须隔离。场景：判断“迁移什么 / 不迁移什么”时优先阅读。
- `docs/integration/sdn_optics_contract.md`：定义 Stage 3 optical module 的最小职责、接口链路和非目标。场景：开始写 optics skeleton 或接口前必须先读。
- `docs/integration/sdn_gap_analysis.md`：整理当前项目需求与 upstream 现状之间的关键差距和风险。场景：拆 issue、评估实现风险和排除误迁移时使用。
- `docs/plan/stage3_sdn_porting_plan.md`：给出 Stage 3A~3F 的执行顺序、issue 拆分和验收边界。场景：安排任务顺序与分支时使用。
- `docs/plan/stage3_contract_freeze.md`：冻结 Stage 3 的正式目标、scope、deliverables 和 non-goals。场景：进入实现前用于边界确认。
- `docs/integration/stage3_unresolved_params.md`：汇总尚未完全拍板的论文 / 实现参数，并区分拍板时点。场景：开 issue 或写实现前先核对 P0 / P1 / P2。

---

## 目录约定
- `configs/`：实验配置，不要把实验参数写死在代码里
- `data/`：数据相关目录，默认不提交大体积原始数据与处理后数据
- `docs/`：论文笔记、复现计划、实验日志、排错文档、结果总结
- `src/`：核心源码，按 datasets / models / metrics / utils / optics 等模块组织
- `scripts/`：训练、评估、数据准备、检查脚本
- `tests/`：小型测试与冒烟测试
- `outputs/`：实验输出，不应作为源码目录使用

除非当前任务明确要求，否则不要随意重构目录结构。

---

## 开发原则
1. 小步修改，围绕单一任务展开  
2. 不要把配置写死在代码中  
3. 数据、模型、训练、评估尽量分离  
4. 不要进行与当前任务无关的大规模重构  
5. 若存在不确定项，应明确写出假设，而不是伪装成确定事实  
6. 优先可读性、可验证性与可追踪性，而非过早优化

---

## 实验与复现原则
1. 先 baseline，后复杂模型  
2. 先验证数据协议与评估协议，再接复杂模块  
3. 先做小规模 sanity check，再做正式训练  
4. 每个实验都应能追溯到配置、代码版本和输出目录  
5. 不默认添加论文未明确说明的技巧；若确实需要假设，应在文档中标注  
6. 如果修改会影响评估协议、数据协议或模型接口，必须显式说明

---

## 文档同步原则
在以下情况下，应该同步建议更新文档：

- 新增重要脚本：更新 `README.md`
- 明确论文细节或发现未明确项：更新 `docs/paper_notes.md`
- 完成阶段性任务：更新 `docs/reproduction_plan.md` 或 issue 状态
- 修复关键错误：更新 `docs/troubleshooting.md`
- 产生重要实验结果：更新 `docs/experiment_log.md` 和 `docs/results_summary.md`

---

## Git 与提交规则
- 默认不要直接改 `main`
- 优先在 `dev` 或 `feature/*` 分支上工作
- 一次修改尽量服务于一个明确任务
- commit message 建议使用：
  - `feat: ...`
  - `fix: ...`
  - `docs: ...`
  - `refactor: ...`
  - `test: ...`
  - `chore: ...`

不要把无关修改混在同一个提交中。

---

## AI agent 输出要求
当你（AI agent）给出代码修改建议时，应尽量做到：

1. 先简要说明准备改什么  
2. 只修改与当前任务相关的文件  
3. 保持改动最小且清晰  
4. 若有假设，显式说明  
5. 若任务超出当前阶段，应提醒而不是直接越阶段实现  
6. 若发现更基础的问题（如数据协议未验证），优先指出，不要盲目堆功能

---

## 禁止事项
除非用户明确要求，否则不要：

- 跳过 baseline 直接上复杂模型
- 把关键参数硬编码到训练脚本中
- 擅自改变项目目录结构
- 把实验输出目录当作源码目录使用
- 用“论文中明确说明”来描述实际上并不确定的细节
- 在没有说明的情况下引入大量额外依赖
