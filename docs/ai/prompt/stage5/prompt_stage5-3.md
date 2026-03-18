You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.3:

Issue title:
Add paper-aligned optics configs for L=1/3/5 (200/400 grid + distances)

This is a Stage-5-scoped optics-configuration task.
Do not turn it into an optical-core rewrite, encoder task, loss task, trainer task, or eval task.

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

Then inspect these implementation-reference files before editing:

7. `src/models/optics/diffractive_decoder.py`
8. `src/models/optics/readout.py`
9. `scripts/check_optical_forward_depths.py`

You may inspect `scripts/train_stage4_minimal.py` only as a narrow reference for
how the current repo instantiates `DiffractiveDecoder`.

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- For unresolved items, obey the ownership rules in `stage5_protocol_freeze.md`.
- If a non-owned unresolved item must be consumed here, keep it configurable and
  clearly mark it as inherited default rather than final truth.

==================================================
1. ISSUE 5.3 GOAL
==================================================

Implement paper-aligned optics configs for `L=1/3/5` on top of the frozen optical contract.

The result should provide a reproducible Stage 5 optics path that:

1. uses `200x200` diffractive layer grids
2. uses `400x400` propagation grids
3. uses explicit `input_to_first / inter_layer / last_to_sensor` distances
4. uses explicit readout crop / FOV settings for the `96x96` target
5. can run forward sanity checks for `L=1/3/5` without changing the optical contract

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not redesign `DiffractiveDecoder`
- do not redefine `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- do not add electrical post-heads or hybrid readout tricks
- do not implement encoder / loss / trainer / eval logic
- do not change dataset tiling rules
- do not finalize phase range policy
- do not finalize gamma or sigma policy
- do not hide distance assumptions in hard-coded depth branches without config visibility
- do not pull Stage 6 quantization / robustness / ablation work into this issue

This issue is successful only if it lands explicit paper-aligned optics configs
and forward-sanity evidence without widening scope.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/optics`

Stay on an optics-oriented branch.
Do not use this issue to modify `feat/model`, `feat/train`, or `feat/eval` concerns.

==================================================
4. REQUIRED INHERITANCE
==================================================

From `stage5_protocol_freeze.md`, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Stage 4 artifacts are regression baseline only.
3. Stage 5 aligns paper settings on top of the current optical core; it does not rewrite that core.
4. Issue 5.3 owns:
   - distance mapping into `input_to_first / inter_layer / last_to_sensor`
   - output crop / FOV alignment details
5. Issue 5.3 does NOT own:
   - final encoder phase range
   - final gamma policy
   - final sigma granularity
   - final `phi_lr` size decision as paper-unique truth

Important inheritance detail:

- If you need `input_pattern_hw` to instantiate Stage 5 optics now, consume the
  current Stage 5 default `32x32` as a configurable inherited default.
- Do not present `32x32` as a paper-uniquely settled fact in this issue.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the Stage 5 config layout for paper-aligned optics
2. the explicit mapping from paper `d1 / d2 / d3` to:
   - `input_to_first`
   - `inter_layer`
   - `last_to_sensor`
3. the explicit `L=1/3/5` distance lists or repeated-distance construction
4. the readout crop / FOV alignment configuration used by Stage 5 optics sanity
5. the exact sanity path used to validate forward behavior for `L=1/3/5`

This issue MUST keep the following out of scope:

1. encoder architecture or phase mapping semantics
2. loss semantics
3. training hyperparameters beyond what is required for config wiring
4. evaluation metrics or bicubic baseline logic

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real optics-config path needed for Stage 5.

Prefer reusing the current abstractions:

- `DiffractiveDecoder`
- `DistanceSchedule`
- `OpticalGridConfig`
- `ReadoutConfig`

The implementation should likely include:

- paper-aligned optics config files under `configs/stage5/` or a clearly
  Stage-5-scoped equivalent under `configs/`
- either reuse of `scripts/check_optical_forward_depths.py` or a new small
  Stage-5-specific sanity script
- explicit comments or a short note explaining the distance mapping and crop policy

Keep the design explicit:

- `layer_hw = (200, 200)`
- `propagation_hw = (400, 400)`
- `output_crop_hw = (96, 96)` unless you discover a documented Stage 5 reason to
  encode crop differently; if so, record it explicitly
- `input_pattern_hw` must remain configurable, even if the current default is `32x32`

For distance mapping, make the implementation auditable.
Do not leave the reader guessing how paper `d1 / d2 / d3` became config values.

==================================================
7. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify that the optics configs are real and inspectable.

At minimum, provide:

- Stage 5 optics config entries for `L=1/3/5`
- a forward-sanity command or script that runs those configs
- concise recorded evidence showing:
  - configured grid sizes
  - configured crop size
  - configured distances for each depth
  - output shapes for `U_out_full`, `I_out_full`, and `I_out_roi`

If the repo already has a preferred place for sanity outputs, follow it.
Do not build a large config framework or experiment manager here.

==================================================
8. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- Stage 5 optics config files under `configs/`
- `scripts/check_optical_forward_depths.py` or a small Stage-5-specific equivalent
- a brief Stage 5 note only if needed to document mapping rationale

Small supporting edits to optics instantiation code are allowed only if they are
strictly necessary to consume explicit configs cleanly.

Avoid:

- broad refactors of the optics stack
- changing encoder modules
- changing trainer / eval entrypoints

==================================================
9. ACCEPTANCE CRITERIA
==================================================

Issue 5.3 is complete only if:

1. Real paper-aligned optics configs exist for `L=1/3/5`
2. Grid, padding, and crop settings are explicit and inspectable
3. Distance mapping is explicit and documented
4. Forward sanity passes for `L=1/3/5` with correct output shapes
5. `phi_lr` size remains configurable if consumed here
6. No optical-core redesign is introduced

==================================================
10. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was implemented
- exact files changed
- the finalized distance mapping for `L=1/3/5`
- where crop / FOV alignment is defined
- what forward-sanity evidence was produced
- what was intentionally left unresolved for later issues
