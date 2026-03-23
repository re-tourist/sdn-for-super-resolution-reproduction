You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 6.1:

Issue title:
Freeze Stage 6 debug protocol and failure taxonomy

This is a Stage-6-scoped documentation task.
Do not write model code, training code, dataset code, or evaluation code in this issue.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
4. `docs/execution/stage5_main_run_report.md`
5. `docs/execution/troubleshooting.md`
6. `docs/plan/stage_plan/stage6/stage6_plan.md`
7. `docs/plan/stage_plan/stage6/stage6_issue_plan.md`

You may read `docs/execution/results_summary.md` only if needed to cross-check one concrete wording claim.
Do not pull in broad old planning docs unless you need a very specific consistency check.

Important scope rule:
- If older docs still describe Stage 6 as a broad systematic-experiments phase,
  follow `stage6_plan.md` and `stage6_issue_plan.md` for current Stage 6 scope.

==================================================
1. ISSUE 6.1 GOAL
==================================================

Write the Stage 6 debug protocol document:

- `docs/plan/stage_plan/stage6/stage6_debug_protocol.md`

This document must freeze the Stage 6 debugging boundary so later Stage 6 implementation
prompts stay narrow, hypothesis-driven, and auditable.

The document must define:

1. why Stage 6 is now debugging-first rather than sweep-first
2. the Stage 6 failure taxonomy
3. the Stage 6 evidence ladder
4. which later issues may change logging only, loss only, or paper-sensitive assumptions
5. what mandatory artifacts every Stage 6 debug run must save

==================================================
2. NON-GOALS — DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not implement eval audit code
- do not implement sigma-aware preview code
- do not implement tiny-set or intervention experiments
- do not reopen Stage 5 protocol directly
- do not silently expand Stage 6 back into quantization / misalignment / modulation sweeps
- do not claim that scale ambiguity is already proven to be the only root cause
- do not write a generic research essay disconnected from later issue execution

This issue succeeds only if it produces a sharp Stage 6 protocol freeze for the next issues.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `docs/update`

Stay on a docs-oriented branch.
Do not use this issue to modify feature branches such as `feat/train`, `feat/eval`, or `feat/optics`.

==================================================
4. REQUIRED INHERITANCE
==================================================

The Stage 6 debug protocol must explicitly inherit these facts:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Stage 5 engineering pipeline already exists and is not being rebuilt from scratch.
3. The current Stage 5 main run is engineering-valid but empirically poor.
4. `Issue T003` established a confirmed scale-ambiguity mechanism.
5. `Issue T003` did NOT prove that scale ambiguity is the only cause of poor performance.
6. Stage 6 must separate:
   - raw intensity collapse
   - normalized reconstruction failure

==================================================
5. WHAT THE DOCUMENT MUST CONTAIN
==================================================

The document should be structured so later Stage 6 prompts can use it directly.

At minimum, include:

### A. Document purpose
- why Stage 6 is being reframed
- what kinds of later issues it constrains

### B. Stage 6 scope boundary
- what belongs to Stage 6 now
- what is explicitly postponed
- explicit rule that Stage 6 is debugging-first, not sweep-first

### C. Failure taxonomy
At minimum define:
- Problem A: raw intensity collapse
- Problem B: normalized reconstruction failure

For each one, clarify:
- typical symptoms
- what evidence is needed
- what kinds of fixes would or would not count as addressing it

### D. Evidence ladder
Define the Stage 6 evidence stages, such as:
- semantics audit
- observability upgrade
- tiny/small-budget reproduction
- narrow intervention
- conditional protocol reopen

### E. Issue authority boundaries
For later Stage 6 issues, state clearly:
- which issues may add logs and diagnostics only
- which issues may run controlled reruns
- which issues may modify loss-like terms
- which issues may reopen paper-sensitive geometry assumptions

### F. Mandatory artifact protocol
Specify what every Stage 6 debug run should record, such as:
- config snapshot
- command
- source checkpoint or baseline reference
- raw and rescaled metrics where applicable
- sigma statistics
- prediction/target sum statistics
- preview artifact expectations
- short report location

### G. Stage-6 exit meaning
- what minimum evidence is needed before broader systematic experiments can resume
- what findings must remain explicit if Stage 6 still ends unresolved

==================================================
6. FAILURE TAXONOMY REQUIREMENTS
==================================================

You must explicitly freeze the distinction between:

1. raw intensity collapse
2. normalized reconstruction failure

And the document must make clear:

- why they are not the same
- why a fix for A does not automatically prove B is solved
- why raw preview must remain a first-class artifact
- why sigma-rescaled or normalized views are diagnostic supplements, not replacements

==================================================
7. ISSUE OWNERSHIP THAT MUST BE MADE EXPLICIT
==================================================

The document must make later-issue ownership clear.
At minimum, it should align later work roughly like this:

- eval semantics audit -> Issue 6.2
- observability / diagnostics -> Issue 6.3
- tiny-set / small-subset reproduction -> Issue 6.4
- scale-fixing interventions -> Issue 6.5
- depth comparison under debug protocol -> Issue 6.6
- paper-sensitive geometry re-audit -> Issue 6.7
- consolidation and next-step policy -> Issue 6.8

Do not leave later issues with ambiguous authority to reopen assumptions.

==================================================
8. STYLE REQUIREMENTS
==================================================

The document should be:

- concise but concrete
- implementation-oriented
- explicit about confirmed mechanisms vs open hypotheses
- explicit about what later issues may and may not do
- useful for future Codex prompts

Avoid:

- vague “we should investigate further” prose without boundaries
- overstating root-cause certainty
- long background sections that do not constrain implementation

==================================================
9. ALLOWED FILE CHANGES
==================================================

Primary target:
- `docs/plan/stage_plan/stage6/stage6_debug_protocol.md`

Optional small companion edits are allowed only if strictly needed for consistency:
- `docs/plan/stage_plan/stage6/stage6_plan.md`
- `docs/plan/stage_plan/stage6/stage6_issue_plan.md`

Do not expand this issue into broader doc cleanup unless a tiny consistency fix is necessary.

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 6.1 is complete only if:

1. `stage6_debug_protocol.md` is created
2. The document clearly states that Stage 6 is debugging-first
3. The document explicitly separates problem A from problem B
4. The document distinguishes confirmed mechanism from open root-cause hypotheses
5. The document defines a clear evidence ladder for later issues
6. The document maps later Stage 6 issue ownership clearly enough that future prompts can say:
   - what this issue may change
   - what this issue must only observe
   - what this issue must not reopen

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was added or updated
- exact files changed
- whether any consistency edits were made outside the new debug protocol doc
- the most important frozen Stage 6 decisions introduced
- the most important open hypotheses intentionally left unresolved
