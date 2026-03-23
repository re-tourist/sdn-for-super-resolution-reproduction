# Stage 6 Debug Protocol Freeze

## 1. Purpose

This document freezes the Stage 6 debugging protocol so later Stage 6
issues stay narrow, hypothesis-driven, and auditable.

Stage 6 is now debugging-first rather than sweep-first because the repo
already has a real Stage 5 paper-aligned pipeline and real execution
artifacts, but the current `phase-only / L=5` main run is still
empirically poor against the bicubic baseline. `Issue T003` established
that scale ambiguity is a confirmed mechanism in the current loss path,
but it did not prove that scale ambiguity is the only root cause of the
poor result.

This protocol directly constrains:

- eval semantics audit
- diagnostics and observability upgrades
- tiny-set and small-subset reruns
- narrow scale-fixing interventions
- depth comparisons
- any later request to reopen paper-sensitive assumptions

If older docs describe Stage 6 as a broad systematic-experiments phase,
follow this document together with
`docs/plan/stage_plan/stage6/stage6_plan.md` and
`docs/plan/stage_plan/stage6/stage6_issue_plan.md`.

## 2. Frozen Inheritance

The following facts are inherited and must remain explicit in all Stage 6
work:

1. The optical contract remains frozen as
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`.
2. The Stage 5 engineering pipeline already exists and is not being
   rebuilt from scratch in Stage 6.
3. The current Stage 5 main run is engineering-valid and audit-ready,
   but empirically poor.
4. `Issue T003` confirmed a real scale-ambiguity mechanism in the current
   Stage 5 path.
5. `Issue T003` did not prove that scale ambiguity is the only cause of
   poor performance.
6. Stage 6 must always separate:
   - raw intensity collapse
   - normalized reconstruction failure

## 3. Scope Boundary

### 3.1 In scope now

Stage 6 currently allows only the following categories of work:

- audit what the existing loss, eval, and preview paths actually mean
- upgrade observability so raw and rescaled evidence can be compared
- reproduce the current `L=5` failure on tiny or small budgets
- run narrow, single-variable, scale-related interventions
- compare `L=3` and `L=5` under the same debug protocol
- reopen paper-sensitive assumptions only after an explicit entry gate

### 3.2 Explicitly postponed

The following remain out of scope unless `Issue 6.8` later re-approves
them:

- broad `L=1/3/5` sweep programs
- quantization studies
- misalignment or robustness matrices
- complex-valued, amplitude-only, or modulation-family comparisons
- broad hyperparameter searches
- silent Stage 5 protocol rewrites disguised as "debugging"

### 3.3 One-sentence rule

Stage 6 debugs the existing Stage 5 paper-path baseline; it does not
resume broad systematic experiments until the current failure is split
into auditable mechanisms with explicit evidence.

## 4. Failure Taxonomy

Stage 6 reports must label the observed failure as one of:

- `A only`
- `B only`
- `A + B`
- `not yet distinguished`

### 4.1 Problem A: raw intensity collapse

Definition:
raw `I_out_roi` intensity is severely collapsed in absolute scale.

Typical symptoms:

- raw `pred_roi` looks near-black under a fixed `[0, 1]` display range
- `prediction_sum << target_sum`
- raw `pred_mean` and `pred_max` are extremely small
- per-sample `sigma` becomes very large because the prediction sum is too
  small

Required evidence:

- raw preview remains saved as a first-class artifact
- `prediction_sum`, `target_sum`, `pred_mean`, `pred_max`
- `sigma` statistics
- raw metrics where applicable

What counts as addressing A:

- raw intensity scale becomes materially less collapsed
- `prediction_sum / target_sum` moves toward a non-degenerate range
- raw previews stop looking trivially black under the fixed display rule
- the report shows that the change affected raw output, not only a
  rescaled diagnostic view

What does not count as addressing A:

- only showing `sigma * pred_roi` or a normalized preview
- only improving a post-rescale visualization while raw output remains
  collapsed
- only arguing from normalized metrics without raw-scale statistics

### 4.2 Problem B: normalized reconstruction failure

Definition:
after explicit, declared rescaling or normalization, the reconstructed
structure is still poor.

Typical symptoms:

- `sigma * pred_roi` still fails to match target structure
- sigma-rescaled or normalized PSNR / SSIM remain poor
- bicubic still clearly outperforms the model after amplitude mismatch is
  accounted for
- line-pair or edge structure remains visibly weak even after declared
  rescaling

Required evidence:

- sigma-rescaled metrics next to raw metrics where applicable
- sigma-rescaled preview and, if used, a clearly labeled normalized
  diagnostic preview
- explicit comparison to the same target and baseline semantics

What counts as addressing B:

- structure improves after a declared intervention under the same eval
  semantics
- sigma-rescaled metrics and visual evidence improve, not just raw
  amplitude
- the report makes clear that the improvement survives after amplitude is
  accounted for

What does not count as addressing B:

- fixing raw amplitude alone
- replacing the raw view with normalized diagnostics
- changing eval semantics and calling the resulting metric gain a model
  improvement without audit evidence

### 4.3 Why A and B are not the same

Problem A is about absolute output scale. Problem B is about structural
reconstruction quality after scale is explicitly handled. A fix for A
does not prove B is solved, and a diagnosis of B does not license hiding
raw-scale failure.

Therefore:

- raw preview must remain a first-class artifact
- sigma-rescaled and normalized views are diagnostic supplements, not
  replacements
- Stage 6 reports must never collapse A and B into a single vague
  statement like "the model is bad"

## 5. Evidence Ladder

Later Stage 6 issues must state which rung they operate on.

### E0. Semantics audit

Goal:
clarify what the current loss, eval, preview, and blind-eval paths
actually measure.

Allowed work:

- code reading
- metric sensitivity checks
- audit notes
- tests or reporting clarifications that expose existing semantics

Not allowed:

- loss changes
- geometry changes
- long reruns justified only by speculation

### E1. Observability upgrade

Goal:
make A and B separable from saved artifacts alone.

Allowed work:

- new logs
- new summary fields
- raw and sigma-rescaled previews
- optional normalized diagnostic preview, clearly labeled as diagnostic
  only

Not allowed:

- loss changes
- silent replacement of the raw preview
- paper-sensitive config changes

### E2. Tiny or small-budget reproduction

Goal:
reproduce the failure with high information density under the frozen
protocol.

Allowed work:

- tiny-set overfit runs
- fixed-budget small-subset reruns
- Stage 6-only debug configs

Not allowed:

- large-budget sweep programs
- geometry reopen
- mixing many unrelated knobs in one run family

### E3. Narrow intervention

Goal:
test one explicit hypothesis at a time against the frozen baseline.

Allowed work:

- one main intervention variable per comparison
- fixed-budget reruns
- explicit baseline anchor

Not allowed:

- bundling unrelated loss and geometry changes
- claiming paper-faithful success from one narrow intervention
- removing the raw-scale evidence requirement

### E4. Conditional protocol reopen

Goal:
reopen paper-sensitive assumptions only if E0 to E3 evidence shows that
scale-related explanations are insufficient.

Entry gate:

- eval semantics already audited
- observability already upgraded or equivalently available
- tiny or small-budget reproduction already recorded
- narrow scale-related interventions still leave major normalized failure
  unexplained

Allowed work:

- minimal re-audit of distance mapping
- minimal re-audit of phase range
- minimal re-audit of crop / FOV alignment

Not allowed:

- wholesale optical redesign
- reopening multiple geometry assumptions at once without evidence

## 6. Issue Authority Boundaries

### 6.1 Logging and diagnostics only

`Issue 6.2` may:

- audit regular eval, blind eval, and metric scale sensitivity
- add minimal audit helpers or tests that expose existing semantics
- produce audit notes under Stage 6 outputs

`Issue 6.2` must not:

- change loss behavior
- reinterpret poor Stage 5 results by silently changing eval semantics
- reopen geometry assumptions

`Issue 6.3` may:

- add logs, previews, summary fields, and report-format upgrades
- save raw and sigma-rescaled evidence side by side
- add normalized diagnostic views only if they stay explicitly secondary

`Issue 6.3` must not:

- change optimization semantics
- hide the raw preview
- change paper-sensitive configs

### 6.2 Controlled reruns

`Issue 6.4` may:

- run tiny-set and small-subset reproductions on the frozen `L=5` path
- add Stage 6 debug configs and reports

`Issue 6.4` must not:

- change loss-like terms
- reopen geometry assumptions
- turn into a large-budget redo

`Issue 6.6` may:

- run controlled `L=3` versus `L=5` debug comparisons
- reuse the same Stage 6 reporting protocol on both depths

`Issue 6.6` must not:

- become a broad depth sweep
- reopen distance mapping unless an earlier gated issue authorizes it

### 6.3 Loss-like terms

Only `Issue 6.5` may modify loss-like or scale-handling assumptions, and
even then only in a narrow, controlled way.

`Issue 6.5` may:

- test weak energy-matching terms
- test nonzero `gamma` for `L=5`
- test an explicit global gain parameter

`Issue 6.5` must not:

- bundle many unrelated loss changes together
- reopen distance mapping, phase range, or crop / FOV assumptions
- declare scale ambiguity to be the only root cause

### 6.4 Paper-sensitive assumptions

Only `Issue 6.7` may reopen paper-sensitive geometry assumptions.

`Issue 6.7` may:

- re-audit distance mapping
- re-audit phase-range assumptions
- re-audit crop / FOV alignment

`Issue 6.7` may start only after:

- `Issue 6.2` is complete
- `Issue 6.4` is complete
- `Issue 6.5` still leaves major normalized failure unexplained

`Issue 6.7` must not:

- redesign the optical core wholesale
- reopen multiple assumptions at once without evidence

### 6.5 Consolidation and policy

`Issue 6.8` may:

- consolidate Stage 6 findings
- update execution summaries
- decide whether broader systematic experiments may resume

`Issue 6.8` must not:

- silently grant new authority to earlier issues
- treat an unresolved Stage 6 outcome as proof of a single root cause

## 7. Mandatory Artifact Protocol

Every Stage 6 debug run must save enough evidence to let a later reader
decide whether it addressed A, B, both, or neither.

### 7.1 Minimum run identity

Each run must record:

- issue id and evidence-ladder rung
- run name and run family
- exact command
- config snapshot
- seed and dataset scope where applicable
- source checkpoint path or explicit baseline reference
- code version reference if available in the existing pipeline

### 7.2 Minimum quantitative evidence

Each run must record, where applicable:

- raw metrics
- sigma-rescaled metrics
- a clear label if any additional normalized metric is diagnostic-only
- `prediction_sum`
- `target_sum`
- `pred_mean`
- `pred_max`
- `sigma` statistics such as `min`, `max`, and `mean`

If a run does not report both raw and sigma-rescaled evidence where the
comparison is meaningful, it cannot claim that A and B were separated.

### 7.3 Minimum preview evidence

Each run must save:

- raw preview of `pred_roi` using the fixed raw display rule
- corresponding target/reference preview
- comparison or error preview where applicable

For blind or structured synthetic targets, each run must also save:

- target grid
- output grid
- comparison grid
- per-target prediction and target sum statistics where applicable

Once observability support exists, each debug run must also save:

- sigma-rescaled preview

Optional only as a supplement:

- per-sample normalized preview, clearly labeled diagnostic-only

Raw preview must never be dropped or replaced by a normalized view.

### 7.4 Minimum report location

Each run must end with a short report or summary artifact under the
Stage 6 reporting path, for example:

- `outputs/stage6/<family>/<run_name>/`
- `docs/execution/`

The report must state:

- whether the run targeted A, B, or both
- what was changed versus the frozen baseline
- what remained frozen
- what conclusion is confirmed
- what remains an open hypothesis

## 8. Stage 6 Exit Meaning

Stage 6 does not exit when "many experiments were tried." It exits when
the current failure has been narrowed into explicit, auditable evidence.

### 8.1 Minimum evidence before broader experiments may resume

Broader systematic experiments remain blocked until Stage 6 has at least:

1. a completed eval-semantics audit
2. observability sufficient to separate A from B in saved artifacts
3. at least one reproducible tiny-set or small-subset characterization
   of the frozen `L=5` failure
4. at least one narrow scale-related intervention result stating whether
   it fixes A only, B only, both, or neither
5. a consolidation decision in `Issue 6.8` that explicitly re-approves
   broader experiment families

### 8.2 If Stage 6 ends unresolved

The final Stage 6 summary must still state explicitly:

- scale ambiguity is confirmed, but not proven to be the only root cause
- whether A was explained
- whether B was explained
- whether depth-specific evidence was found
- whether any paper-sensitive assumption was reopened and why
- which experiment families remain blocked after Stage 6

## 9. Prompt Contract For Later Issues

Any later Stage 6 implementation prompt should state explicitly:

- target problem: `A`, `B`, or `A + B`
- evidence rung: `E0` to `E4`
- allowed changes
- forbidden changes
- mandatory artifacts to save

If a prompt cannot answer those five items, it is too broad for the
frozen Stage 6 protocol.
