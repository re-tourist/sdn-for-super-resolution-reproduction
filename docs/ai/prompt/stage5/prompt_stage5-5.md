You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.5:

Issue title:
Add paper-aligned SR loss with efficiency penalty

This is a Stage-5-scoped loss task.
Do not turn it into an optics rewrite, trainer framework task, or eval task.

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
8. `configs/stage5/stage5_paper_encoder.yaml`

Then inspect these implementation-reference files before editing:

9. `scripts/train_stage4_minimal.py`
10. `scripts/train_decoder_only_single_sample.py`
11. `scripts/train_decoder_only_small_subset.py`

Important scope rules:

- If `paper_notes.md` is broader than the Stage 5 schedule, follow the Stage 5 docs.
- For unresolved items, obey the ownership rules in `stage5_protocol_freeze.md`.
- The Stage 4 `normalized_mae_roi(...)` implementations are historical reference only.
  They are not automatically the final Stage 5 loss contract.

==================================================
1. ISSUE 5.5 GOAL
==================================================

Implement the paper-aligned Stage 5 loss:

- normalized MAE on `I_out_roi`
- plus an optional efficiency penalty term

The result should provide a reproducible Stage 5 loss path that:

1. consumes prediction and target tensors on the ROI path
2. computes `sigma` according to the Stage 5 loss contract
3. computes the efficiency term with an explicit `eta` definition
4. keeps `gamma` configurable by depth
5. keeps unresolved policy choices explicit and auditable

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not modify optics geometry or readout semantics
- do not modify encoder semantics
- do not build the Stage 5 trainer here
- do not build eval scripts here
- do not add Stage 6 ablation logic
- do not hard-code an unreviewed `L=5` gamma as if it were paper-settled fact
- do not silently change target supervision from `I_out_roi` to another tensor
- do not broaden this into a generic repo-wide loss framework unless a very small
  shared helper is strictly justified

This issue is successful only if it lands the Stage 5 paper-aligned loss
contract in a narrow, configurable, auditable way.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/train`

Stay on a training/loss-oriented branch.
Do not use this issue to modify `feat/model` or `feat/eval` concerns.

==================================================
4. REQUIRED INHERITANCE
==================================================

From `stage5_protocol_freeze.md`, this issue must inherit and obey:

1. Frozen optical contract:
   `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
2. Main supervision target remains `I_out_roi`
3. Stage 4 artifacts are regression baseline only
4. Issue 5.5 owns:
   - final `L=5 gamma` policy for Stage 5
   - final `sigma` normalization granularity for Stage 5
   - the Stage 5 efficiency-term config interface
5. Issue 5.5 does NOT own:
   - dataset protocol
   - distance mapping
   - phase range
   - crop/FOV semantics
   - trainer architecture

Important inheritance detail:

- `5.3` and `5.4` already froze the Stage 5 optics/encoder path.
- Your loss must consume those paths, not reopen them.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the Stage 5 loss module path and API
2. the explicit `sigma` computation mode used by Stage 5
3. the explicit `eta` computation interface used by Stage 5
4. the `gamma_by_depth` defaults, including `L=5`
5. the config fields used to enable/disable the efficiency penalty

This issue MUST keep the following explicit and auditable:

1. `sigma` definition
2. `eta` definition
3. `gamma_by_depth`
4. whether the efficiency term is enabled
5. whether these are Stage 5 defaults or stronger conclusions

This issue MUST keep the following out of scope:

1. trainer loop changes beyond tiny loss-call integration helpers
2. eval metric code
3. any Stage 6 sweep or robustness logic

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Implement the smallest real loss path needed for Stage 5.

Prefer a narrow, explicit structure such as:

- a new module under `src/losses/` if that directory does not yet exist
- a clearly named Stage-5-specific loss class/function
- a small config example under `configs/stage5/`
- a small sanity script under `scripts/` only if needed

The implementation should likely include:

- normalized MAE with explicit `sigma` normalization
- optional efficiency penalty term using:
  - output power `P_o`
  - input power `P_i`
  - `eta = 100 * P_o / P_i`
- configurable `gamma_by_depth`
- an explicit switch for turning the efficiency term on/off

You must keep the interface practical for later Stage 5 trainer work.
For example, the loss path may need to consume:

- `prediction_roi`
- `target_roi`
- optional extra tensors needed for power accounting, if the efficiency term requires them
- current depth or preselected gamma value

Do not bury the required inputs in hidden globals.

==================================================
7. REFERENCE MATH TO IMPLEMENT
==================================================

The Stage 5 target loss contract is:

`L = mean_i | y_i - sigma * y_hat_i | + gamma * exp(-eta)`

with:

`sigma = sum_i y_i / (sum_i y_hat_i + epsilon)`

and:

`eta = 100 * P_o / P_i`

You must make explicit:

- over which dimensions `sigma` is computed
- how batch aggregation is performed
- what tensors are used for `P_o` and `P_i`
- what happens when the efficiency term is disabled

==================================================
8. REQUIRED ARTIFACTS
==================================================

You must produce enough artifacts to verify that the loss path is real and inspectable.

At minimum, provide:

- the Stage 5 loss module
- config entries for:
  - `epsilon`
  - `sigma_mode` or equivalent
  - `gamma_by_depth`
  - efficiency-term enable/disable
- a fake-batch or small sanity command/script showing:
  - loss is finite
  - `sigma` is finite
  - `eta` is finite when enabled
  - toggling the efficiency term changes the total loss in a controlled way

If the repo already has a preferred place for Stage 5 sanity outputs, follow it.
Do not build a heavy training harness here.

==================================================
9. FILE SCOPE
==================================================

Primary files likely to be changed or added:

- `src/losses/` for the new loss implementation
- `configs/stage5/` for the loss config
- a small sanity script under `scripts/` only if needed

Avoid:

- broad trainer changes
- optics changes
- encoder changes
- eval changes

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 5.5 is complete only if:

1. A real Stage 5 paper-aligned loss path exists
2. `sigma` normalization is explicit and documented
3. Efficiency penalty behavior is explicit and configurable
4. `gamma_by_depth` is configurable and logged
5. `L=5` gamma handling is no longer left implicit
6. Loss runs stably on a fake batch without NaN/Inf

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was implemented
- exact files changed
- the chosen Stage 5 `sigma` policy
- the chosen Stage 5 `gamma_by_depth` policy, including `L=5`
- how `eta` is computed in the implementation
- what sanity evidence was produced
- what was intentionally left unresolved for later issues
