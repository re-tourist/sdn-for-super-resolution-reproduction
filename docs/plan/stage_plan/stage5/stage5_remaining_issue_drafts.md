# Stage 5 收口 Issue 草案

## 1. 文档定位

本文档不是重新拆分全部 Stage 5，而是基于当前仓库状态，为 **尚未真正收口** 的 Stage 5 issue 提供可直接粘贴到 GitHub 的标题和正文草案。

当前判断依据来自：

- `docs/ai/PROJECT_CONTEXT.md`
- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
- `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
- `docs/execution/stage5_smoke_report.md`
- `docs/execution/stage5_main_run_report.md`
- `docs/run_order.md`
- `docs/execution/experiment_log.md`
- `docs/execution/results_summary.md`

截至当前状态：

- `5.1 ~ 5.9` 已有明确协议、实现或运行证据
- `5.10` 的 launch path 已准备，但真实 Linux-server 主运行证据仍待补齐
- `5.11` 尚未完成，且承担 Stage 5 收口与 Stage 6 入口判定责任

---

## 2. Issue 5.10

Title:

`Run paper-aligned main training (phase-only, L=5) and write report`

Body:

```md
## Background
Stage 5.1~5.9 have already frozen the protocol and landed the paper-aligned
dataset, optics config, encoder, loss, trainer, regular eval, blind eval, and
smoke validation path.

The remaining Stage 5 main task is no longer “make the path exist”.
The path already exists. The remaining work is to execute the real `L=5`
phase-only main run on the Linux server and record the evidence honestly.

## Suggested Branch
`feat/train`

## Goal
Execute the Stage 5 paper-aligned `L=5` main run using the already prepared
launch path, then fill the main-run report with actual run evidence.

## Current State
- `configs/stage5/stage5_trainer_main.yaml` already exists
- `docs/run_order.md` already contains the Linux-server train / resume / eval commands
- `docs/execution/stage5_main_run_report.md` already exists as a result template
- `docs/execution/stage5_smoke_report.md` already shows smoke-level readiness

This issue should consume those artifacts rather than redesigning them.

## Tasks
- Launch the phase-only `L=5` main training on the Linux server using the prepared Stage 5 main config
- If the run is resumed, record the exact resume command and checkpoint path
- Run regular eval on both `val` and `test` using the existing Stage 5 eval path
- Run blind line-pair eval using the existing Stage 5 blind-eval path
- Fill `docs/execution/stage5_main_run_report.md` with:
  - run identity
  - actual commands used
  - output roots
  - completed steps / stop point
  - best checkpoint information
  - val / test metrics
  - blind eval summary
  - remaining gaps
- If the run stops before the full target budget, record the exact stop point and reason explicitly

## Resource Boundary
- Distinguish clearly between:
  - code/config path is ready
  - long-run compute budget was or was not fully available
- A partial, interrupted, or budget-limited run is still useful evidence, but it must not be reported as if it were a full-budget completed run
- Do not treat the earlier smoke run as main-run evidence

## Non-Goals
- Do not rewrite `scripts/train_stage5_paper.py`
- Do not redesign `scripts/eval_stage5_paper.py` or `scripts/eval_stage5_blind_linepair.py`
- Do not reopen Stage 5 protocol defaults
- Do not add Stage 6 sweeps, quantization, or robustness work here

## Deliverable
- Main training outputs under `outputs/stage5/main_run/`
- Regular eval outputs under `outputs/stage5/eval/`
- Blind eval outputs under `outputs/stage5/blind_eval/`
- Completed `docs/execution/stage5_main_run_report.md`

## Acceptance
- A real Linux-server `L=5` paper-aligned main run is executed and logged
- Val / test / blind-eval results are recorded in a comparable format
- The report clearly separates implementation readiness from resource-limited execution state
- No smoke-only evidence is misrepresented as main-run evidence

## Reference Docs
- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
- `docs/execution/stage5_smoke_report.md`
- `docs/execution/stage5_main_run_report.md`
- `docs/run_order.md`
```

---

## 3. Issue 5.11

Title:

`Consolidate Stage 5 results and make Stage-6 go/no-go decision`

Body:

```md
## Background
Stage 5 is only useful if its planning docs, implementation evidence, and run
outputs are consolidated into one authoritative repo state.

Right now, several execution-facing docs still reflect the earlier Stage 4 or
Stage-5-planning-only state. After Issue 5.10, the repo needs one explicit
close-out pass that states:
- what Stage 5 actually completed
- what remained resource-limited
- whether Stage 6 may start

## Suggested Branch
`docs/update`

## Goal
Consolidate Stage 5 execution evidence and make an explicit Stage 6 `GO / NO-GO`
decision with reasons.

## Dependency
This issue should start after Issue 5.10 has produced either:
- a completed main-run record, or
- a clearly documented budget-limited stop point with real evidence

## Tasks
- Update `docs/execution/experiment_log.md` so it no longer stops at Stage 4
- Update `docs/execution/results_summary.md` so it reflects the actual Stage 5 state rather than “planning only”
- Review `docs/execution/stage5_smoke_report.md` and `docs/execution/stage5_main_run_report.md` and summarize what they prove
- Record which Stage 5 assumptions remained defaults and which were actually exercised by the real run
- Classify remaining gaps into:
  - engineering gap
  - compute/resource gap
  - paper-ambiguity gap
- Make an explicit Stage 6 decision:
  - `GO`
  - `conditional GO`
  - `NO-GO`
- If the decision is `conditional GO` or `NO-GO`, list the exact blockers and the issue family that should resolve them
- Link the approved Stage 6 plan so the next phase starts from a frozen document rather than ad-hoc discussion

## Resource Boundary
- Do not hide missing long-run budget inside vague wording like “results pending”
- If Stage 5 is engineering-complete but budget-limited, say that directly
- If Stage 5 is still technically incomplete, say that directly instead of blaming compute by default

## Non-Goals
- Do not reopen Stage 5 implementation scope
- Do not silently start Stage 6 experiments inside this issue
- Do not rewrite historical reports to claim evidence that was never produced

## Deliverable
- Updated `docs/execution/experiment_log.md`
- Updated `docs/execution/results_summary.md`
- Any necessary updates to:
  - `docs/execution/troubleshooting.md`
  - `docs/paper/paper_notes.md`
  - `docs/plan/stage_plan/stage6/stage6_plan.md`
- A clear Stage 6 decision recorded in repo docs

## Acceptance
- The repo states clearly what Stage 5 completed and what it did not complete
- Stage 5 evidence is consolidated instead of being scattered across planning-only and execution-only documents
- Stage 6 entry status is explicit, justified, and linked to concrete evidence
- Remaining blockers, if any, are written as actionable next work rather than vague uncertainty

## Reference Docs
- `docs/execution/stage5_smoke_report.md`
- `docs/execution/stage5_main_run_report.md`
- `docs/execution/experiment_log.md`
- `docs/execution/results_summary.md`
- `docs/plan/stage_plan/stage6/stage6_plan.md`
```

---

## 4. 使用建议

如果只准备继续当前主线，建议先发：

1. `Issue 5.10`
2. `Issue 5.11`

原因：

- `5.10` 负责把 Stage 5 从“launch-ready”推进到“有真实主运行证据”
- `5.11` 负责把仓库状态从“文档分散”推进到“可以正式进入 Stage 6”

在 `5.11` 完成之前，不建议把 Stage 6 执行 issue 大规模开出来，因为那样很容易绕开 Stage 5 的最后收口。
