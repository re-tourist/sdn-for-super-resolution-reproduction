You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.2:

Issue title:
Build paper-aligned 96x96 EMNIST display dataset with augmentation

This is a Stage-5-scoped data-protocol task.
Do not write optics code, encoder code, loss code, trainer code, or evaluation code in this issue.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/paper/paper_notes.md`
4. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
5. `docs/plan/stage_plan/stage5/stage5_plan.md`
6. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`

You may inspect current Stage 4 dataset / trainer code only as implementation reference.
Do not treat Stage 4 data protocol as the target to preserve. It is only a baseline reference.

Important scope rule:
- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- For unresolved items, respect the ownership defined in `stage5_protocol_freeze.md`.

==================================================
1. ISSUE 5.2 GOAL
==================================================

Implement the paper-aligned EMNIST display dataset path for Stage 5.

The result should provide a reproducible dataset pipeline that:

1. starts from EMNIST letters
2. resizes 28x28 letters to 32x32
3. composes 96x96 display samples
4. enforces the train / val / test sample-count protocol
5. enforces the train/val vs test letter-count distribution
6. exposes augmentation options consistent with the paper
7. saves enough preview artifacts to audit the protocol

==================================================
2. NON-GOALS — DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not implement optics configs
- do not implement encoder / loss / trainer / eval code
- do not introduce natural-image datasets
- do not redesign Stage 4 data code unless a tiny shared helper is strictly justified
- do not add Stage 6 quantization / robustness / ablation logic
- do not silently bury tiling rules or split logic inside code without logging/config

This issue is successful only if it cleanly freezes and implements the Stage 5 data protocol.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/data`

Stay on a data-oriented branch.
Do not use this issue to modify `feat/optics`, `feat/model`, or `feat/train`.

==================================================
4. REQUIRED INHERITANCE
==================================================

From `stage5_protocol_freeze.md`, this issue must inherit and obey:

1. Stage 5 is the first paper-aligned stage.
2. Stage 4 artifacts are regression baseline only.
3. The optical contract remains frozen, but this issue must not touch it.
4. The `96x96` display tiling rule is owned by Issue 5.2 and may be finalized here.
5. Other unresolved items such as distance mapping, phase range, gamma policy, and crop alignment are NOT owned by this issue and must not be finalized here.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the exact 96x96 tiling rule
2. how many 32x32 cells are used and how they are occupied
3. how train/val/test letter-count distributions are instantiated
4. how dataset seeds / randomness are controlled
5. how augmentation options are exposed in config

This issue MUST keep the following out of scope:

1. optics geometry
2. phase mapping
3. gamma policy
4. sigma normalization policy
5. output crop / FOV alignment

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement a Stage 5 dataset module and the minimum surrounding support needed to inspect it.

The implementation should likely include:

- a dataset module under `src/data/` or the repo's current equivalent
- config entries for:
  - dataset root
  - split
  - sample count
  - letter-count distribution
  - RNG seed
  - augmentation toggles
- a preview / sanity path to save generated samples

Prefer a deterministic and auditable design:

- same seed -> same split / same sample generation behavior
- protocol choices visible in config snapshot or summary
- preview artifacts easy to inspect manually

==================================================
7. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify that the dataset protocol is real and inspectable.

At minimum, provide:

- a config example or config section for Stage 5 dataset construction
- a small preview artifact showing generated 96x96 samples
- a concise summary of:
  - split sizes
  - train/val letter-count range
  - test letter-count range
  - seed / randomization policy
  - augmentation policy

If the repo already has a preferred place for saved previews or summaries, follow it.
Do not build a heavy dataset visualization framework.

==================================================
8. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- dataset code under `src/data/`
- Stage 5 dataset config under `configs/`
- a small preview / sanity script under `scripts/` only if strictly needed
- a brief execution or design note only if needed to explain finalized tiling rules

Avoid unrelated edits.
Do not refactor the entire data layer.

==================================================
9. ACCEPTANCE CRITERIA
==================================================

Issue 5.2 is complete only if:

1. A real Stage 5 dataset path exists for EMNIST -> 96x96 display samples
2. Train / val / test protocol is explicit and reproducible
3. The finalized tiling rule is visible in code/config and not hidden
4. Preview artifacts make the generated protocol auditable
5. Augmentation options are exposed and limited to the Stage 5 paper-aligned scope
6. No non-owned unresolved item is silently finalized here

==================================================
10. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was implemented
- exact files changed
- the finalized 96x96 tiling rule
- how split generation and randomness are controlled
- what preview artifacts were produced
- what was intentionally left unresolved for later issues
