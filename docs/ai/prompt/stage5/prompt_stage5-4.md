You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.4:

Issue title:
Implement paper-aligned phase-only encoder for Stage 5

This is a Stage-5-scoped encoder task.
Do not turn it into an optics-core rewrite, loss task, trainer task, or eval task.

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
7. `configs/stage5/stage5_optics_paper_aligned.yaml`

Then inspect these implementation-reference files before editing:

8. `src/models/encoders/minimal_phase_encoder.py`
9. `src/models/hybrid/minimal_hybrid_wrapper.py`
10. `src/models/optics/diffractive_decoder.py`
11. `src/models/optics/phase_provider.py`

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- For unresolved items, obey the ownership rules in `stage5_protocol_freeze.md`.
- This issue owns the final Stage 5 encoder-facing decision for:
  - exposed `phi_lr` spatial size
  - phase mapping range
- Even if you finalize those for Stage 5, do not present them as paper-unique
  truth unless the documents actually support that claim.

==================================================
1. ISSUE 5.4 GOAL
==================================================

Implement a paper-aligned phase-only encoder that maps `96x96` HR input to a
phase-domain `phi_lr` tensor compatible with the frozen optical contract and
the Stage 5 optics configs from Issue 5.3.

The result should provide a reproducible Stage 5 encoder path that:

1. accepts single-channel `96x96` HR input
2. outputs phase-only `phi_lr`
3. matches the Stage 5 optics input pattern size
4. applies an explicit phase-range mapping on the encoder side
5. integrates with the current `DiffractiveDecoder.forward_from_phase(...)`
   path without modifying optical-core semantics

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not redesign `DiffractiveDecoder`
- do not change the frozen contract
  `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- do not move `phi_lr -> U0` into the encoder
- do not output field-domain tensors from the encoder
- do not implement Stage 5 loss / trainer / eval logic
- do not reopen distance mapping or crop/FOV alignment from Issue 5.3
- do not change dataset protocol or tiling rules from Issue 5.2
- do not add Stage 6 ablations, quantization, or robustness logic
- do not make broad framework refactors just to fit the encoder

This issue is successful only if it lands a real Stage 5 encoder module with
explicit phase semantics and narrow integration.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/model`

Stay on a model-oriented branch.
Do not use this issue to modify `feat/train` or `feat/eval` concerns.

==================================================
4. REQUIRED INHERITANCE
==================================================

From `stage5_protocol_freeze.md`, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. `phi_lr -> U0` remains owned by the optical decoder.
3. Stage 4 artifacts are regression baseline only.
4. Issue 5.3 already froze the Stage 5 optics config path, including:
   - `input_pattern_hw`
   - distance mapping
   - crop / FOV alignment
5. Issue 5.4 owns:
   - the encoder-exposed `phi_lr` size used by Stage 5
   - the Stage 5 phase mapping range
6. Issue 5.4 does NOT own:
   - distance mapping
   - loss semantics
   - gamma policy
   - sigma normalization policy
   - eval protocol

Important inheritance detail:

- The current Stage 5 optics config uses `input_pattern_hw = [32, 32]`.
- Your encoder must integrate with that path cleanly.
- If you keep `32x32` as the Stage 5 encoder output, record it as the chosen
  Stage 5 encoder/optics alignment, not as paper-unique truth unless the docs
  explicitly justify that stronger claim.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the Stage 5 encoder module path and interface
2. the Stage 5 default `phi_lr` output size
3. the Stage 5 default phase mapping range
4. the config fields used to expose `target_hw` and `phase_range`
5. the narrow integration path between encoder output and current decoder input

This issue MUST keep the following explicit and auditable:

1. the chosen `phi_lr` size
2. the chosen phase mapping range
3. whether those are Stage 5 defaults, inherited defaults, or stronger conclusions

This issue MUST keep the following out of scope:

1. optics geometry changes
2. loss implementation
3. trainer architecture
4. evaluation scripts

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real encoder path needed for Stage 5.

Prefer reusing existing repo structure:

- encoder code under `src/models/encoders/`
- narrow hybrid wiring under `src/models/hybrid/` only if strictly needed
- Stage 5 config entries under `configs/stage5/`

The implementation should likely include:

- a new paper-aligned encoder module under `src/models/encoders/`
- export wiring in `src/models/encoders/__init__.py`
- a small Stage 5 encoder config example or config section
- a narrow sanity / integration script, or a self-check path, that proves:
  - `96x96` input is accepted
  - `phi_lr` has the intended shape
  - `phi_lr` stays in the configured phase range
  - the current decoder can consume it successfully

If `src/models/hybrid/minimal_hybrid_wrapper.py` is too Stage-4-specific to host
the new encoder cleanly, you may make a narrow supporting edit or add a new
small Stage-5-specific wrapper. Do not generalize the whole model stack.

Keep phase semantics explicit:

- raw encoder features/logits may exist for debugging
- the mapped output handed to the decoder must be real-valued phase-domain `phi_lr`
- phase mapping must be visible in code and config

==================================================
7. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify that the encoder path is real and inspectable.

At minimum, provide:

- the Stage 5 encoder module
- config entries for encoder target size and phase range
- a sanity command or script that exercises encoder + decoder integration
- concise recorded evidence showing:
  - encoder input shape
  - encoder output `phi_lr` shape
  - configured phase range
  - observed `phi_lr` min/max or equivalent range check
  - successful downstream decoder forward shapes

If the repo already has a preferred place for saved Stage 5 sanity outputs,
follow it. Do not build a large model-debug framework here.

==================================================
8. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- `src/models/encoders/` for the new encoder
- `src/models/encoders/__init__.py`
- `src/models/hybrid/` only if a narrow integration helper is needed
- Stage 5 config under `configs/stage5/`
- a small sanity script under `scripts/` only if needed

Avoid:

- broad refactors of the optics stack
- trainer changes
- loss changes
- eval changes

==================================================
9. ACCEPTANCE CRITERIA
==================================================

Issue 5.4 is complete only if:

1. A real Stage 5 paper-aligned encoder path exists
2. The encoder outputs phase-domain `phi_lr`, not field-domain tensors
3. Output shape matches the Stage 5 optics input pattern contract
4. Phase mapping range is explicit, configurable, and logged
5. Encoder output integrates with the current diffractive decoder forward path
6. No optics-core redesign is introduced

==================================================
10. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was implemented
- exact files changed
- the chosen Stage 5 `phi_lr` size
- the chosen Stage 5 phase mapping range
- what sanity / integration evidence was produced
- what was intentionally left unresolved for later issues
