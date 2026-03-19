# Stage 5.11 反馈

本次 Issue 5.11 没有新增训练、评估或模型实现，而是把已经存在的
Stage 5 执行证据收口到 repo 的摘要层文档里，并明确给出阶段结论。

## 本次完成的核心结论

- Stage 5 工程完成度结论：`PASS`
- 论文级质量复现结论：`NOT YET DEMONSTRATED`
- Stage 6 进入决策：`GO`

这个结论的含义是：

- Stage 5 的 paper-aligned pipeline 现在已经真实存在并有完整 artifact
- 真实 smoke run、真实 `phase-only / L=5` main run、matching regular eval、
  matching blind eval 都已经落地
- 但当前模型在已报告的 `val/test` PSNR 和 SSIM 上仍然落后于 bicubic
- blind eval 目前仍然是 artifact-first 证据，不应被包装成更强的
  blind-resolution 成功声明

## 本次更新的文件

- `docs/execution/results_summary.md`
- `docs/execution/experiment_log.md`
- `docs/ai/PROJECT_CONTEXT.md`
- `docs/ai/CodexFeedback/stage5/feedback_stage5-11.md`

## 我在摘要层文档里固定下来的判断

### 1. Stage 5 工程层面已经完成

摘要文档现在明确写清：

- dataset / optics / encoder / loss / trainer / regular eval / blind eval
  全部已经落地
- `docs/execution/stage5_smoke_report.md` 是真实 smoke 证据
- `docs/execution/stage5_main_run_report.md` 是真实主训练证据
- `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`
  共同构成了 Stage 5 的执行证据闭环

### 2. Stage 5 经验层面不能夸大

摘要文档现在明确保留这些 caveat：

- main run 虽然真实完成，但不能据此宣称论文结果已经复现
- bicubic 在当前 `val/test` 报告里都优于模型
- blind eval 有真实 artifact，但没有 paper-final scalar blind metric
- Stage 6 的 systematic follow-up 还没有开始执行

### 3. Stage 6 可以进入，但理由是“工程就绪”

`GO` 的理由不是“结果已经很好”，而是：

- 当前 repo 已经有可信的 paper-aligned 起点
- 已经不再缺 dataset / optics / encoder / loss / trainer / eval 基础设施
- 后续问题已经从“能不能跑通”转成“为什么当前质量落后于 bicubic，
  应该如何做系统化排查”

## 最重要的经验 caveat

- `val`:
  - model `PSNR = 14.9181`
  - model `SSIM = 0.8126`
  - bicubic `PSNR = 25.6875`
  - bicubic `SSIM = 0.9448`
- `test`:
  - model `PSNR = 9.5774`
  - model `SSIM = 0.4386`
  - bicubic `PSNR = 20.2487`
  - bicubic `SSIM = 0.8393`

所以 Stage 5 的正确表述必须是：

- 工程完成：是
- 论文级质量：否
- 进入 Stage 6：可以

## 推荐的 commit 三段式信息

```text
docs(execution | stage5-11): consolidate stage5 verdict and stage6 entry

why:
close out stage5 with an explicit repo-level verdict so the project
records what the paper-aligned pipeline actually demonstrated, what it
did not demonstrate, and whether stage6 should begin from the current
artifact base

what:
update the summary-layer docs to record stage5 engineering completion as
pass, preserve the main-run and eval evidence, state that bicubic still
outperforms the model on reported PSNR/SSIM, and mark stage6 entry as go
for systematic follow-up work
```
