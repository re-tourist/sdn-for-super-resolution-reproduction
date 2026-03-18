You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.1:

Issue title:
Freeze Stage 5 paper-aligned protocol and unresolved-parameters ledger

This is a Stage-5-scoped documentation task.
Do not write model code, training code, dataset code, or evaluation code in this issue.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/paper/paper_notes.md`
4. `docs/plan/stage_plan/stage4/stage4_protocol_freeze.md`
5. `docs/plan/stage_plan/stage5/stage5_plan.md`
6. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`

You may read `docs/plan/plan_overview.md` only if needed for a wording or boundary cross-check.
Do not pull in large execution logs unless you need to verify one concrete claim.

Important scope rule:
- If `paper_notes.md` records a wider reproduction roadmap than the current Stage 5 schedule,
  follow `stage5_plan.md` and `stage5_issue_plan.md` for stage scope.

==================================================
1. ISSUE 5.1 GOAL
==================================================

Write the Stage 5 protocol freeze document:

- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`

This document must freeze the Stage 5 paper-aligned implementation boundary so later
implementation prompts can be narrow, non-drifting, and auditable.

The document must define:

1. what Stage 5 is trying to align
2. what Stage 5 explicitly does not include
3. what Stage 3 / Stage 4 facts are inherited and frozen
4. what paper-aligned defaults are selected for Stage 5
5. what remains unresolved
6. which later issue is allowed to finalize each unresolved item

==================================================
2. NON-GOALS — DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not implement dataset code
- do not implement optics configs
- do not implement encoder / loss / trainer / eval code
- do not rewrite Stage 4 history
- do not promote Stage 6 work into Stage 5
- do not silently resolve paper ambiguities as settled fact
- do not create a broad design manifesto unrelated to later issue execution

This issue is successful only if it produces a sharp protocol freeze for later work.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `docs/update`

Stay on a docs-oriented branch.
Do not use this issue to modify feature branches such as `feat/train`, `feat/model`, or `feat/optics`.

==================================================
4. REQUIRED INHERITANCE
==================================================

The Stage 5 protocol freeze must explicitly inherit these facts:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Stage 4 learnability gate has already passed.
3. Stage 4 artifacts are the regression baseline, not work to redo.
4. Stage 5 is the first stage that aligns to the paper's full settings.
5. Stage 6 remains the home for:
   - quantization sweep
   - misalignment / robustness
   - systematic L=1/3/5 ablations
   - complex-valued / amplitude-only comparisons

==================================================
5. WHAT THE DOCUMENT MUST CONTAIN
==================================================

The document should be structured so that later implementation issues can use it directly.

At minimum, include:

### A. Document purpose
- what this freeze is for
- what kinds of later issues it constrains

### B. Scope boundary
- in-scope Stage 5 items
- out-of-scope Stage 6 items
- explicit conflict-resolution rule:
  Stage 5 planning docs override wider paper-roadmap phrasing when stage scope differs

### C. Inherited frozen contracts
- optical contract
- Stage 4 regression baseline role
- no optical-core redesign

### D. Paper-aligned target protocol
- dataset protocol
- optics geometry targets
- encoder / phase representation target
- loss / gamma policy
- training hyperparameter targets
- eval target protocol

### E. Unresolved parameters ledger
For each unresolved item, specify:
- current default handling
- what is still unknown
- which later issue may finalize it
- what later issues must not silently finalize it

### F. Issue constraints
Add short constraints for later issue families:
- dataset
- optics config
- encoder
- loss
- trainer
- eval
- smoke / main run

### G. Acceptance / exit criteria
- what must exist before Stage 5 is considered engineering-complete
- what stronger evidence is resource-dependent

==================================================
6. UNRESOLVED ITEMS THAT MUST BE HANDLED CAREFULLY
==================================================

You must explicitly address at least these unresolved items:

1. `phi_lr` size
2. 96x96 display tiling rule
3. distance mapping into `input_to_first / inter_layer / last_to_sensor`
4. L=5 gamma policy
5. phase mapping range
6. sigma normalization granularity
7. output crop / FOV alignment details

For each one:
- do not pretend it is fully settled if it is not
- assign ownership to a later issue
- make clear whether later implementation must keep it configurable

==================================================
7. STYLE REQUIREMENTS
==================================================

The document should be:

- concise but concrete
- implementation-oriented
- explicit about assumptions
- explicit about forbidden drift
- useful for future Codex prompts

Avoid:

- generic research prose
- repeating the paper at length
- long background sections that do not constrain implementation

==================================================
8. ALLOWED FILE CHANGES
==================================================

Primary target:
- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`

Optional small companion edits are allowed only if strictly helpful for consistency:
- `docs/plan/stage_plan/stage5/stage5_plan.md`
- `docs/plan/stage_plan/stage5/stage5_issue_plan.md`

But do not expand the task into broader doc refactoring unless a tiny consistency fix is necessary.

==================================================
9. ACCEPTANCE CRITERIA
==================================================

Issue 5.1 is complete only if:

1. `stage5_protocol_freeze.md` is created
2. The document clearly separates Stage 5 from Stage 6
3. The document explicitly inherits the frozen optical contract
4. The document records all major unresolved items honestly
5. Each unresolved item is mapped to a later issue owner
6. The document is specific enough that later implementation prompts can say:
   - what this issue may finalize
   - what this issue must keep configurable
   - what this issue must not touch

==================================================
10. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was added or updated
- exact files changed
- whether any consistency edits were made outside the new freeze doc
- the most important frozen decisions introduced
- the most important unresolved items left intentionally open
