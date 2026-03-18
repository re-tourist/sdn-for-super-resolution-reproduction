You are working in a research engineering repository for reproducing
“Super-resolution image display using diffractive decoders”.

Your task is to complete GitHub Issue 4.7:

Issue title:
Run small-subset closed-loop training and produce first Stage-4 summary

This issue starts only after Issue 4.6 has already passed.
Do NOT re-open or re-implement the single-sample acceptance work.
Use the existing Stage 4 training stack as-is unless a tiny run-blocking fix is strictly necessary.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these:

- stage4_issue_plan.md
- stage4_plan.md
- stage4_protocol_freeze.md
- feedback_stage4-6.md

Also inspect the current repo implementation that Issue 4.7 depends on, especially:
- `scripts/train_stage4_minimal.py`
- the current Stage 4 encoder / wrapper / dataset path / trainer-related files already used in Issue 4.6
- any report/output conventions used by the repo

Important context from Issue 4.6:
- single-sample closed-loop acceptance already PASSed
- encoder gradients were observed
- optics gradients were observed
- no NaN/Inf
- the next narrow step is the small-subset acceptance run
Do not turn this issue into a redesign of trainer / loss / optics / dataset. :contentReference[oaicite:6]{index=6}

==================================================
1. ISSUE 4.7 GOAL
==================================================

Run the first small-subset Stage 4 closed-loop training acceptance experiment and summarize whether the learnability observed in single-sample overfit extends beyond one sample.

This issue is about acceptance evidence, not architectural expansion.

You must answer, with real run artifacts:

1. Does loss still decrease on a small real subset?
2. Do encoder and optics gradients remain observable?
3. Are outputs input-dependent across samples?
4. Is there any obvious collapse to a single pattern?
5. Does the evidence support a PASS or FAIL judgment for small-subset Stage 4 acceptance?

==================================================
2. NON-GOALS — DO NOT DRIFT
==================================================

Do NOT do any of the following unless absolutely required by a run-blocking bug:

- do not redesign the optical core
- do not redesign the dataset / adapter / wrapper
- do not introduce Stage 5 paper-aligned settings
- do not add large-scale training
- do not rewrite the trainer into a general framework
- do not perform L=1/3/5 sweeps
- do not introduce quantization / robustness / misalignment studies
- do not “improve” Issue 4.6 retroactively
- do not create stage-based source directories

==================================================
3. BRANCH / WORKTREE EXPECTATION
==================================================

This issue belongs to the training / acceptance path.

Preferred working branch:
- `feat/train`

If `feat/train` is no longer clean or has diverged too much from latest `dev`,
create a clean branch from latest `dev` such as:
- `feat/train-stage4`

Do NOT work on `docs/*` branches.

Before coding/running, verify the current branch and working tree are appropriate.

==================================================
4. WHAT TO RUN
==================================================

Use the existing Stage 4 trainer and run a small-subset closed-loop acceptance experiment.

Stay close to the existing single-sample pipeline from Issue 4.6.
Reuse:
- same trainer
- same dataset path style
- same model stack
- same artifact conventions

Select a small real subset size appropriate for Stage 4 acceptance.
A reasonable target is something like:
- 8 / 16 / 32 samples

Choose a small subset size that is:
- non-trivial
- computationally manageable
- sufficient to expose whether the model collapses or remains input-dependent

Do not silently choose a huge run.

==================================================
5. REQUIRED ARTIFACTS
==================================================

The run output directory should contain the usual Stage 4 trainer artifacts, including as applicable:

- config snapshot
- history
- run summary
- loss curve
- representative previews
- phase previews
- gradient statistics
- checkpoints

In addition, for this issue you MUST ensure there is enough evidence to inspect:
- multiple different inputs
- their corresponding outputs
- whether outputs differ across inputs
- whether there is obvious collapse

If the current trainer already saves suitable multi-sample previews, use that.
If it does not, add only the smallest necessary artifact/reporting enhancement to support this issue.

Do NOT turn this into a visualization framework overhaul.

==================================================
6. REQUIRED ANALYSIS
==================================================

You must inspect and summarize at least:

### A. Loss behavior
- initial vs best vs final loss
- whether train/val loss decrease meaningfully on the subset

### B. Gradient observability
- whether encoder gradients are non-zero across training
- whether optics gradients are non-zero across training

### C. Output diversity / anti-collapse check
- whether outputs differ across different inputs
- whether the model collapses to a single generic bright blob / generic pattern
- whether the learned behavior remains input-dependent

### D. Numerical stability
- NaN / Inf presence or absence
- obvious instability or not

### E. Honest verdict
Give a clear PASS / FAIL judgment for Issue 4.7 specifically.

==================================================
7. REPORT DELIVERABLE
==================================================

Create or update the concise report:

- `docs/execution/stage4_small_subset_report.md`

The report must include:

1. Task boundary
2. Run command(s)
3. Device and artifact directory
4. Core scalar results
5. Preview / artifact inspection findings
6. Gradient findings
7. Stability findings
8. PASS / FAIL verdict
9. If PASS, what remains unresolved but non-blocking
10. The most natural next step after Issue 4.7

The tone must be honest and engineering-oriented.
Do not exaggerate success.
Do not pretend Stage 5 quality has been reached.

==================================================
8. ALLOWED CODE CHANGES
==================================================

Preferred path:
- run the existing trainer with no code change if possible

If a small-subset acceptance run reveals that one tiny change is strictly required to:
- select a small subset
- save representative multi-sample previews
- report gradients / summaries cleanly

then make only the narrowest necessary change.

Do NOT do unrelated refactors.

==================================================
9. ACCEPTANCE CRITERIA
==================================================

Issue 4.7 is complete only if:

1. A real small-subset closed-loop run has been executed
2. Loss decreases meaningfully on the subset
3. Encoder gradients remain observable
4. Optics gradients remain observable
5. There is evidence that outputs are input-dependent
6. There is no obvious collapse to a single pattern
7. `docs/execution/stage4_small_subset_report.md` gives an honest PASS / FAIL judgment
8. The issue remains Stage-4-scoped and does not drift into Stage 5

==================================================
10. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of what was run / changed
- exact files changed
- exact run command(s)
- artifact output path(s)
- core results summary
- PASS / FAIL verdict for Issue 4.7
- any remaining non-blocking caveats