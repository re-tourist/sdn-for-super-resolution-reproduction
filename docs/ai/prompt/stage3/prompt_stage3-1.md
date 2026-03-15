Task: Record finalized Stage-3 design decisions into project documentation.

Context
-------
Stage-3 planning documents were generated earlier:

- docs/integration/stage3_unresolved_params.md
- docs/plan/stage3_contract_freeze.md
- AGENTS.md

Those documents intentionally listed several "Immediate Decisions" that required
human confirmation before starting optical implementation.

The human review has now finalized those decisions.

Your task is NOT to redesign anything.
Your task is to **record the finalized decisions clearly in the documentation**
so that future development agents can rely on them.

Do NOT modify any source code.
Do NOT introduce new architecture.
Only update documentation.

---------------------------------------------------------------------

Human-approved Stage-3 decisions
--------------------------------

Decision 1 — Layer semantics

Stage-3 defines **L as the number of trainable diffractive phase masks**.

- L ∈ {1, 3, 5}
- Final propagation to the sensor plane is NOT counted as a layer
- Propagation distances must be controlled by an explicit distance schedule
- No implicit layer-index special cases (e.g. `layer == 2`)

Decision 2 — Distance representation

Propagation distances must use **ordered list with explicit semantics**.

Configuration must explicitly distinguish:

- input_to_first
- inter_layer distances
- last_to_sensor

Example schema (conceptual):

distance_schedule:
  input_to_first: ...
  inter_layer: [...]
  last_to_sensor: ...

This schema is now frozen for Stage-3.

Decision 3 — ROI crop contract

Stage-3 adopts **explicit center-crop ROI extraction**.

Pipeline must be:

phi_lr
 → U0
 → U_out_full
 → I_out_full
 → I_out_roi (center crop)
 → loss / metrics

Crop must NOT be hidden in scripts.

Use configuration parameters such as:

output_crop_hw

The exact crop indexing logic must be deterministic.

Decision 4 — Supervision contract

Stage-3 uses:

**full-grid forward + ROI supervision**

Meaning:

- optical propagation produces `I_out_full`
- loss and main metrics are computed ONLY on `I_out_roi`
- `I_out_full` is retained for visualization, diagnostics, and physics checks

Decision 5 — Normalized MAE implementation

Stage-3 normalized MAE default implementation:

sigma = sum(target) / sum(pred)

Computed per sample on ROI pixels.

Batch aggregation:
mean over samples.

Numerical protection:
epsilon added to denominator.

This is the **Stage-3 engineering default**.
Further paper-exact alignment may happen in later stages.

---------------------------------------------------------------------

Required documentation updates
------------------------------

1️⃣ Update  
`docs/integration/stage3_unresolved_params.md`

- Mark the five "Immediate Decisions" as **RESOLVED**
- Move them to a section like:

    "Finalized Stage-3 Decisions"

- Keep explanation concise
- Remove wording that implies they are still undecided

2️⃣ Update  
`docs/plan/stage3_contract_freeze.md`

Ensure the contract section explicitly reflects:

- L semantics
- distance schedule schema
- ROI crop contract
- ROI supervision rule
- normalized MAE default

Do NOT expand scope of Stage-3.

3️⃣ Update  
`AGENTS.md`

Add a short note under the Stage-3 planning section:

"Immediate design decisions have been finalized.
Future agents must follow the rules defined in:

- stage3_contract_freeze.md
- stage3_unresolved_params.md"

This ensures coding agents read these rules before implementing optics modules.

---------------------------------------------------------------------

Constraints
-----------

- Do NOT modify project architecture
- Do NOT change Stage-3 scope
- Do NOT modify source code
- Only update documentation clarity and decision status

---------------------------------------------------------------------

Output summary

After editing, report:

- which files were modified
- what sections were updated
- confirmation that the five decisions are now recorded as finalized