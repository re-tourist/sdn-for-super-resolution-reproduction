You are working in a research engineering repository for reproducing
“Super-resolution image display using diffractive decoders”.

Your task is to implement the Stage 4 minimal phase encoder in a way that is fully consistent with the latest Stage 4 planning documents and the repo’s modular architecture requirements.

This prompt supersedes any earlier suggestion that used stage-specific source directories such as `src/models/stage4/`.
Do NOT create or use stage-based directories under `src/`.

==================================================
0. MUST-READ FILES FIRST
==================================================

Before writing any code, you MUST carefully read and follow these documents:

- stage4_issue_plan.md
- stage4_plan.md
- stage4_protocol_freeze.md

These three files are the source of truth for:
- Stage 4 scope
- Stage 4 non-goals
- tensor contract
- minimal closed-loop definition
- implementation priorities

Key facts from those docs that you must obey:
- Stage 4 is only about proving end-to-end learnability, not paper-final alignment. 
- The frozen optical mainline is:
  `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- The Stage 4 minimal closed loop is:
  `HR input -> minimal encoder -> phi_lr -> optical decoder -> I_out_roi -> loss`
- Current Stage 3 toy/synthetic scripts are not Stage 4 proof.
- Stage 4 must not redesign the optical core.
- Stage 4 must use modular code organization, not stage-specific source directories. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

Also inspect the real repo code that this issue depends on, especially:
- `src/models/optics/diffractive_decoder.py`
- `scripts/train_electronic_baseline.py`
- any nearby model modules needed for naming/style consistency

==================================================
1. ARCHITECTURE CONSTRAINT — VERY IMPORTANT
==================================================

Do NOT create:
- `src/models/stage4/`
- or any other stage-based source-code directory

Reason:
Stage is an experiment/protocol dimension, not a source-code architecture dimension.

Use a modular structure instead.

Preferred target path:
- `src/models/encoders/minimal_phase_encoder.py`

If needed, you may also add:
- `src/models/encoders/__init__.py`

But do NOT touch unrelated directories.

==================================================
2. WHAT THIS ISSUE IS REALLY ABOUT
==================================================

Implement the first minimal learned encoder for Stage 4.

Its job is ONLY:
- take a single-channel HR image tensor as input
- output a phase-domain tensor `phi_lr`
- be easy to debug
- be compatible with the frozen optical decoder contract

This is NOT:
- a paper-faithful encoder reproduction
- a trainer
- a dataset pipeline
- a hybrid wrapper
- a config overhaul
- a Stage 5 feature

==================================================
3. FROZEN CONTRACT YOU MUST OBEY
==================================================

The encoder must fit the Stage 4 tensor contract:

Input:
- `x_hr`: single-channel HR image
- shape semantics: `[B, 1, H_hr, W_hr]`

Output:
- `phi_lr`: phase-domain tensor
- shape semantics: `[B, 1, H_phi, W_phi]`

Requirements:
- `phi_lr` is NOT a generic latent tensor
- `phi_lr` must already have phase semantics before entering the optical decoder
- the raw-to-phase mapping belongs to the encoder side, not the optical decoder side
- the exact global Stage 4 sizes are not all fully frozen yet, so keep spatial size configurable

Do not hardcode paper-final dimensions unless the current repo already enforces them.

==================================================
4. IMPLEMENTATION REQUIREMENTS
==================================================

Create a minimal encoder module under:

- `src/models/encoders/minimal_phase_encoder.py`

Define a class such as:
- `MinimalPhaseEncoder`

It should include at least:

### A. Clear constructor
The constructor should expose:
- input channel count (default 1)
- target output spatial size, e.g. `target_hw`
- base/hidden channel count
- phase range, defaulting to something symmetric such as `(-pi, pi)`

### B. Minimal CNN backbone
Keep it intentionally small and readable.
A reasonable design is:
- a few Conv + activation blocks
- then resize/downsample to `target_hw`
- then output one channel
- then map raw output explicitly into phase range

Prefer simplicity and debuggability over cleverness.

### C. Explicit phase mapping
You MUST implement an explicit raw-to-phase mapping in code.
For example, a bounded mapping based on tanh is acceptable.

The mapping should be easy to inspect and should not be hidden.

### D. Shape validation
Add lightweight validation where reasonable:
- input must be `[B, 1, H, W]`
- output channel must be 1
- target size behavior should be obvious

### E. Good docstrings
Docstrings must explicitly say:
- this is a Stage 4 minimal encoder
- it outputs phase-domain `phi_lr`
- it is not the final paper-aligned encoder
- stage should be represented by protocol/config, not source directory structure

==================================================
5. DESIGN PREFERENCES
==================================================

Use the current repo style as much as possible:
- type hints where appropriate
- readable code
- no unnecessary abstraction
- no giant framework

Good engineering choices:
- keep `target_hw` configurable
- keep `phase_range` configurable
- make the mapping helper explicit
- optionally expose a method like `encode_raw_phase(...)` if it helps debugging

Do NOT:
- modify the optical core just to accommodate the encoder
- reimplement optical logic
- add a hybrid wrapper in this issue
- add trainer code in this issue
- create stage-named source directories

==================================================
6. OPTIONAL SELF-CHECK
==================================================

If repo style allows, add a very small smoke-level self-check helper or minimal usage example.
But keep it tiny.
Do NOT create a large test harness in this issue.

The goal is only to make the module easier to inspect.

==================================================
7. ACCEPTANCE CRITERIA
==================================================

The issue is complete only if:

1. `src/models/encoders/minimal_phase_encoder.py` exists
2. it defines a minimal encoder class for Stage 4
3. it consumes `[B, 1, H_hr, W_hr]`
4. it produces phase-domain `phi_lr` with shape `[B, 1, H_phi, W_phi]`
5. the phase range mapping is explicit in code
6. the implementation is modular and does NOT use `src/models/stage4/`
7. the code is clearly compatible in principle with `DiffractiveDecoder.forward_from_phase(...)`
8. the code remains intentionally minimal and non-paper-final

==================================================
8. FILE CHANGE BOUNDARY
==================================================

Allowed:
- `src/models/encoders/minimal_phase_encoder.py`
- `src/models/encoders/__init__.py` if needed

Avoid editing anything else unless it is strictly necessary for import hygiene.

If you believe another file must be changed, keep it minimal and justify it clearly.

==================================================
9. FINAL RESPONSE FORMAT
==================================================

After making changes, respond with:

- short summary of what was implemented
- exact files changed
- encoder input/output contract
- chosen phase-range mapping
- any assumptions or deferred choices that remain
- confirmation that no `src/models/stage4/` directory was created or used