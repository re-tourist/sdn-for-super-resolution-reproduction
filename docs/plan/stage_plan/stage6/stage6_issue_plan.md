# 第六阶段 Issue 规划（Stage 6 — 系统性排错、语义审计与总结）

下面给出一份可直接拆成 GitHub issues 的 Stage 6 工程规划。

Stage 6 的主目标不是做大规模 sweep，而是解释当前 Stage 5 主线为什么失败，并用最小、最可解释的证据把问题收敛。范围以 `docs/plan/stage_plan/stage6/stage6_plan.md` 为准。

范围提醒：

- Stage 6 当前是 **systematic debugging**，不是 **systematic experiments**。
- quantization、misalignment、complex-valued 大对照等内容，除非被 `Issue 6.8` 重新批准，否则不应提前恢复。
- 任何 Stage 6 implementation prompt 都应显式区分：
  - 问题 A：raw intensity collapse
  - 问题 B：normalized reconstruction failure

---

## Issue 6.1 — Freeze Stage 6 debug protocol and failure taxonomy

Title:
`Freeze Stage 6 debug protocol and failure taxonomy`

Body:
```md
## Background
Stage 5 engineering has landed, but the main paper-aligned result is still empirically poor. Before adding more experiments, Stage 6 must freeze a debugging protocol that defines what counts as confirmed mechanism, open hypothesis, and acceptable intervention scope.

## Suggested Branch
`docs/update`

## Goal
Write and freeze a Stage-6 debugging protocol that defines:
- the two main failure classes:
  - raw intensity collapse
  - normalized reconstruction failure
- the evidence ladder for Stage 6
- which issues may change logging only, loss only, or paper-sensitive configs
- the minimum artifact/reporting fields for all Stage 6 runs

## Tasks
- Review `stage5_protocol_freeze.md`, `stage5_main_run_report.md`, and `troubleshooting.md`
- Freeze the Stage-6 failure taxonomy
- Freeze the Stage-6 evidence ladder
- Define mandatory Stage-6 reporting fields
- Define which later issues are allowed to reopen which assumptions

## Deliverable
- `docs/plan/stage_plan/stage6/stage6_debug_protocol.md`

## Acceptance
- Stage 6 scope is clearly debugging-first rather than sweep-first
- Raw-collapse and normalized-structure failure are separated explicitly
- Later implementation issues can tell what they may or may not change
```

---

## Issue 6.2 — Audit regular/blind eval semantics and metric scale sensitivity

Title:
`Audit regular/blind eval semantics and metric scale sensitivity`

Body:
```md
## Background
The current main run shows raw outputs near black, yet raw and sigma-rescaled metrics remain almost identical on audit subsets. Before more training, we must verify whether eval semantics preserve or hide amplitude information.

## Suggested Branch
`feat/eval`

## Goal
Audit the regular eval path, blind eval path, and metric helpers to answer whether PSNR/SSIM and blind reporting are sensitive to global output scale.

## Frozen Inheritance
- Keep Stage-5 crop / FOV semantics unchanged during audit
- Reuse the existing eval scripts and metric helpers as the audit target
- Do not silently fix Stage 5 behavior while auditing it

## Tasks
- Inspect `src/eval/metrics.py`, `src/eval/evaluator.py`, `scripts/eval_stage5_paper.py`, and `scripts/eval_stage5_blind_linepair.py`
- Make normalization, `data_range`, and crop assumptions explicit
- Add a simple scale-sensitivity audit such as comparing `pred` vs `c * pred`
- Record whether bicubic and model paths receive any asymmetric normalization
- Write a short audit note or summary artifact

## Non-Goals
- Do not change loss behavior here
- Do not launch new long training runs here
- Do not redesign the Stage-5 eval runner

## Deliverable
- Eval audit note and/or summary artifact under `outputs/stage6/metric_audit/`
- Minimal code or test updates needed to make semantics explicit

## Acceptance
- It is clear whether regular metrics are raw-amplitude-sensitive
- It is clear whether blind eval preserves or hides amplitude information
- Any hidden normalization or invariance is documented explicitly
```

---

## Issue 6.3 — Add sigma-aware diagnostics and preview/reporting upgrades

Title:
`Add sigma-aware diagnostics and preview/reporting upgrades`

Body:
```md
## Background
The current previews show raw `pred_roi`, which is useful, but Stage 6 also needs sigma-aware and normalized diagnostic views to separate intensity collapse from structure failure.

## Suggested Branch
`feat/scripts`

## Goal
Upgrade Stage-5/Stage-6 training and eval reporting so every debug run exposes raw, sigma-rescaled, and key scale statistics.

## Frozen Inheritance
- Keep raw preview as a first-class artifact
- Keep Stage-5 tensor contract unchanged
- Only add observability; do not change optimization semantics here

## Tasks
- Add preview support for raw `pred_roi`
- Add preview support for `sigma * pred_roi`
- Optionally add per-sample normalized preview as a diagnostic-only view
- Log `prediction_sum`, `target_sum`, `sigma`, `pred_max`, and `pred_mean`
- Log raw and sigma-rescaled metrics side by side where applicable

## Non-Goals
- Do not modify the loss function here
- Do not reinterpret Stage-5 results by hiding the raw view
- Do not add broad experiment logic here

## Deliverable
- Updated preview/reporting path
- Diagnostic artifacts under `outputs/stage6/diagnostics/`

## Acceptance
- A Stage-6 debug run can separate problem A from problem B using saved artifacts alone
- Raw preview remains available
- Scale statistics are written to summaries in a reproducible format
```

---

## Issue 6.4 — Reproduce the L=5 failure on tiny-set and small-subset budgets

Title:
`Reproduce the L=5 failure on tiny-set and small-subset budgets`

Body:
```md
## Background
Before reopening physics assumptions, we need high-information-density reruns that tell us whether the current failure already exists on tiny data, or only appears at larger scale.

## Suggested Branch
`feat/train`

## Goal
Build tiny-set and small-subset debug runs for the frozen `L=5 phase-only` path and characterize whether the failure is scale-only, optimization-related, or already structural on tiny data.

## Frozen Inheritance
- Use the frozen Stage-5 paper-aligned path as the anchor
- Reuse Stage-5 dataset, optics, encoder, and eval semantics unless explicitly overridden by earlier audit findings
- Keep changes confined to Stage-6 debug configs and reports

## Tasks
- Add tiny-set debug configs (for example 1 to 4 samples)
- Add fixed-budget small-subset configs
- Run overfit / short-run diagnostics on the frozen `L=5` path
- Report raw vs rescaled metrics and sigma traces
- Record whether raw output, normalized output, or both fail

## Non-Goals
- Do not start a large-budget main-run redo here
- Do not compare many unrelated hyperparameters here
- Do not modify paper-sensitive geometry here

## Deliverable
- Debug configs under `configs/stage6/`
- Tiny-set / small-subset reports under `docs/execution/`
- Artifacts under `outputs/stage6/tinyset/` and `outputs/stage6/smallsubset/`

## Acceptance
- The repo has a reproducible tiny-set failure characterization for `L=5`
- It is clear whether tiny-set can fit raw intensity, normalized structure, both, or neither
- Results are recorded as evidence, not as ad hoc observations
```

---

## Issue 6.5 — Test narrow scale-fixing interventions on the frozen L=5 path

Title:
`Test narrow scale-fixing interventions on the frozen L=5 path`

Body:
```md
## Background
Scale ambiguity is now a confirmed mechanism. Stage 6 should test a few narrow, hypothesis-driven interventions to see whether fixing scale also improves empirical quality.

## Suggested Branch
`feat/train`

## Goal
Evaluate a small set of tightly scoped scale-fixing interventions on the frozen `L=5` path.

## Allowed Intervention Types
- weak energy matching term
- nonzero `gamma` policy for `L=5`
- optional explicit global gain parameter

## Tasks
- Choose a minimal intervention set and make it configurable
- Keep a frozen Stage-5-style baseline as the anchor
- Run fixed-budget comparisons only
- Report which interventions improve:
  - raw intensity only
  - normalized structure only
  - both
  - neither

## Non-Goals
- Do not bundle many unrelated loss changes together
- Do not change optics geometry here
- Do not declare any intervention to be the final paper-faithful solution without evidence

## Deliverable
- Narrow debug configs under `configs/stage6/scale_fix/`
- Comparison report under `docs/execution/`
- Artifacts under `outputs/stage6/scale_fix/`

## Acceptance
- At least one controlled scale-fixing comparison is completed
- The report distinguishes "fixes problem A" from "fixes problem B"
- The intervention scope remains narrow and auditable
```

---

## Issue 6.6 — Compare L=3 and L=5 under the same debug protocol

Title:
`Compare L=3 and L=5 under the same debug protocol`

Body:
```md
## Background
We still need to know whether the current failure is specifically worse at `L=5`, or whether it reflects a broader problem in the paper-aligned path.

## Suggested Branch
`feat/train`

## Goal
Run a controlled debug comparison between `L=3` and `L=5` using the same Stage-6 protocol and observability fields.

## Frozen Inheritance
- Use the same Stage-6 audit and reporting protocol on both depths
- Keep dataset and eval semantics fixed
- Reuse the same reporting fields introduced earlier in Stage 6

## Tasks
- Prepare comparable `L=3` and `L=5` debug configs
- Run fixed-budget debug jobs under the same protocol
- Compare raw intensity, sigma distribution, and normalized metrics
- Conclude whether the failure is strongly depth-specific or more general

## Non-Goals
- Do not reopen distance mapping here unless earlier issues require it
- Do not expand this into a full `L=1/3/5` paper trend study
- Do not mix in quantization or robustness experiments here

## Deliverable
- Controlled depth-debug artifacts under `outputs/stage6/depth_debug/`
- Short comparison report under `docs/execution/`

## Acceptance
- The repo has a direct `L=3` vs `L=5` debug comparison
- The comparison uses the same diagnostic protocol on both runs
- Depth-specific vs depth-agnostic failure is stated explicitly
```

---

## Issue 6.7 — Re-audit paper-sensitive geometry assumptions only if needed

Title:
`Re-audit paper-sensitive geometry assumptions only if needed`

Body:
```md
## Background
Only after eval semantics and scale-related failure are audited should Stage 6 revisit paper-sensitive geometry assumptions such as distance mapping, phase range, or crop alignment.

## Suggested Branch
`feat/optics`

## Goal
Perform a narrow re-audit of paper-sensitive geometry assumptions if earlier Stage-6 evidence shows that scale-related explanations are insufficient.

## Entry Gate
This issue should only start if:
- Issue 6.2 is complete
- Issue 6.4 is complete
- Issue 6.5 still leaves major normalized-structure failure unexplained

## Tasks
- Re-audit distance mapping against Stage-5 protocol freeze
- Re-audit phase-range assumptions and encoder mapping
- Re-audit crop / FOV alignment assumptions end to end
- If a change is proposed, document it as a minimal and explicit protocol deviation

## Non-Goals
- Do not redesign the optical core wholesale
- Do not restart from scratch with a new paper interpretation
- Do not change multiple geometry assumptions at once without evidence

## Deliverable
- Narrow physics-audit note under `docs/execution/` or `docs/plan/stage_plan/stage6/`
- Minimal configs/artifacts under `outputs/stage6/physics_audit/`

## Acceptance
- Any reopened assumption is justified by earlier Stage-6 evidence
- The scope of any geometry change is explicit and minimal
- If no change is justified, the issue documents that conclusion clearly
```

---

## Issue 6.8 — Consolidate Stage 6 debugging findings and redefine post-Stage-6 policy

Title:
`Consolidate Stage 6 debugging findings and redefine post-Stage-6 policy`

Body:
```md
## Background
Stage 6 is only useful if its findings are consolidated into a clear statement of what failed, what was explained, and whether the repo may return to broader experiments.

## Suggested Branch
`docs/update`

## Goal
Summarize Stage-6 debugging evidence, update the core execution docs, and decide what categories of experiments are allowed next.

## Tasks
- Update `docs/execution/troubleshooting.md`
- Update `docs/execution/results_summary.md`
- Update `docs/execution/experiment_log.md`
- Summarize which failure mechanisms are confirmed, rejected, or still open
- State whether the repo may resume broader systematic experiments, and under what constraints

## Deliverable
- Updated execution docs
- A Stage-6 summary note with a clear next-step policy

## Acceptance
- The repo clearly distinguishes:
  - engineering baseline status
  - raw intensity collapse status
  - normalized reconstruction status
  - eval semantics status
  - next approved experiment families
- Future work is constrained by explicit evidence rather than open-ended trial and error
```

---

## 推荐执行顺序

1. `6.1`
2. `6.2`
3. `6.3`
4. `6.4`
5. `6.5`
6. `6.6`
7. `6.7`
8. `6.8`

---

## 推荐 Milestone

`Stage 6 — Systematic Debugging`
