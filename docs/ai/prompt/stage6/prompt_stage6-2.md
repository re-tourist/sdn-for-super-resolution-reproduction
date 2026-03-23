You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 6.2:

Issue title:
Audit regular/blind eval semantics and metric scale sensitivity

This is a Stage-6-scoped audit task.
It may include small eval-side code, test, or reporting changes if needed to expose
existing semantics clearly, but it is NOT a loss-change or training-change issue.

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
8. `docs/plan/stage_plan/stage6/stage6_debug_protocol.md`
9. `src/eval/metrics.py`
10. `src/eval/evaluator.py`
11. `scripts/eval_stage5_paper.py`
12. `scripts/eval_stage5_blind_linepair.py`
13. `configs/stage5/stage5_eval.yaml`
14. `configs/stage5/stage5_blind_eval.yaml`

You may inspect related summary artifacts only if needed to verify one concrete claim.
Do not pull in broad unrelated logs.

Important scope rule:
- This issue audits semantics first.
- Do not silently “fix” Stage 5 behavior before documenting what the current behavior actually is.

==================================================
1. ISSUE 6.2 GOAL
==================================================

Determine, with explicit evidence, what the current regular eval path and blind eval path
actually measure.

This issue must answer at least these questions:

1. Are current `PSNR / SSIM` calculations sensitive to global output scale?
2. Does any part of the metric path normalize, clip, or otherwise wash out amplitude information?
3. Are bicubic and model predictions treated symmetrically before scoring?
4. Does blind eval preserve raw amplitude information, or only provide shape-oriented artifacts?
5. Is the current near-equality of raw vs sigma-rescaled metrics better explained by:
   - metric-path invariance or normalization
   - clipping behavior
   - or the model remaining structurally poor even after rescaling

==================================================
2. NON-GOALS — DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not change the Stage 5 loss
- do not change training behavior
- do not reopen optics geometry, distance mapping, or phase range
- do not redesign the Stage 5 eval runner
- do not rewrite bicubic policy for performance reasons
- do not launch new long training runs
- do not claim a root cause beyond what the audit evidence supports

If you discover a problem, document it precisely.
Do not turn this issue into a broad “fix everything” refactor.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `feat/eval`

Keep this issue centered on eval semantics and audit evidence.

==================================================
4. REQUIRED INHERITANCE
==================================================

The audit must explicitly inherit these facts:

1. Stage 5 paper-path baseline is frozen and must remain the anchor.
2. `Issue T003` confirmed scale ambiguity, but did not prove it is the only root cause.
3. Stage 6 debug protocol requires separating:
   - Problem A: raw intensity collapse
   - Problem B: normalized reconstruction failure
4. This issue is mainly an `E0` semantics-audit issue under the Stage 6 evidence ladder.

==================================================
5. WHAT YOU MUST AUDIT
==================================================

You must audit all of the following explicitly.

### A. `src/eval/metrics.py`

Check and document:

- shape preparation behavior
- any normalization or clipping path
- `data_range` assumptions
- whether metric helpers are invariant or approximately invariant to global scaling
- whether values greater than `1.0` are clipped before scoring

### B. `src/eval/evaluator.py`

Check and document:

- whether it adds any extra semantics beyond `metrics.py`
- whether batch aggregation hides per-sample behavior that matters for the audit

### C. `scripts/eval_stage5_paper.py`

Check and document:

- how model predictions are passed into metrics
- how bicubic predictions are built and scored
- whether model and bicubic are normalized differently
- whether summary fields state the current semantics clearly enough

### D. `scripts/eval_stage5_blind_linepair.py`

Check and document:

- whether blind eval preserves raw prediction statistics
- whether any normalization/clipping is applied before saved artifacts
- whether the blind path is shape-focused, amplitude-preserving, or mixed

### E. Synthetic scale-sensitivity checks

You must run at least one minimal, explicit audit using simple synthetic tensors, such as:

- `pred`
- `c * pred`
- possibly `pred` values crossing `1.0` after scaling

The goal is to show how `compute_psnr` / `compute_ssim` respond under current helper semantics.

==================================================
6. DELIVERABLES
==================================================

At minimum, produce:

1. A short audit document, for example:
   - `docs/execution/stage6_eval_semantics_audit.md`

2. A reproducible audit artifact under Stage 6 outputs, for example:
   - `outputs/stage6/metric_audit/<run_name>/summary.json`

3. If useful, a small helper script and/or focused test, such as:
   - `scripts/audit_stage6_eval_semantics.py`
   - `tests/eval/test_metric_scale_sensitivity.py`

You may also make minimal code or summary-field edits in existing eval files if that is the
smallest way to expose current semantics clearly.

==================================================
7. ALLOWED CHANGES VS FORBIDDEN CHANGES
==================================================

Allowed:

- audit notes
- synthetic audit scripts
- focused tests
- small clarifying edits in eval code or summaries
- making current normalization/clipping semantics explicit

Forbidden:

- changing loss-like behavior
- changing training configs
- changing optics configs
- silently changing Stage 5 evaluation meaning and then presenting new metrics as comparable

If you must change runtime code, keep the behavior either:

- fully unchanged but better exposed, or
- explicitly marked as an audit-only clarification with before/after semantics documented

==================================================
8. ACCEPTANCE CRITERIA
==================================================

Issue 6.2 is complete only if:

1. The repo has a clear written answer to whether current PSNR/SSIM are raw-amplitude-sensitive.
2. Any hidden normalization, clipping, or implicit value-range handling is documented explicitly.
3. The audit states whether bicubic and model paths are scored symmetrically.
4. The audit states whether blind eval preserves amplitude information or only shape-oriented artifacts.
5. The audit distinguishes:
   - metric-path semantics
   - raw-intensity collapse
   - normalized reconstruction quality
6. The result is narrow enough that `Issue 6.3` can build directly on it without reopening ambiguity.

==================================================
9. STYLE REQUIREMENTS
==================================================

The result should be:

- specific
- audit-oriented
- evidence-based
- careful not to overclaim causality
- useful for later Stage 6 prompts

Avoid:

- generic prose about “further investigation”
- large unrelated refactors
- changing meanings without documenting them

==================================================
10. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was audited
- exact files changed
- exact files added
- the most important metric/eval semantics discovered
- whether any runtime behavior was changed or only documented
- what Stage 6 should treat as confirmed after this audit
- what still remains open for `Issue 6.3` and later issues
