You are working in a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is to complete GitHub Issue 5.11:

Issue title:
Consolidate Stage 5 results and make Stage-6 go/no-go decision

This is a Stage-5-scoped documentation and consolidation task.
Do not turn it into new model/trainer/eval implementation work.

==================================================
0. MUST-READ CONTEXT FIRST
==================================================

Before doing anything, you MUST read and follow these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
4. `docs/plan/stage_plan/stage5/stage5_plan.md`
5. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`
6. `docs/execution/stage5_smoke_report.md`
7. `docs/execution/stage5_main_run_report.md`
8. `docs/execution/results_summary.md`
9. `docs/execution/experiment_log.md`

Then inspect these evidence-bearing files before editing:

10. `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`
11. `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
12. `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`
13. `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`
14. `docs/ai/CodexFeedback/stage5/feedback_stage5-10.md`

Important scope rules:

- This issue consolidates already-existing Stage 5 facts.
- Do not create new experimental claims without evidence in the existing artifacts.
- Do not rewrite history to make the current results sound better than they are.

==================================================
1. ISSUE 5.11 GOAL
==================================================

Consolidate the now-complete Stage 5 execution evidence into the repo's
summary-layer documents and make an explicit Stage 6 go/no-go decision.

The result should clearly answer:

1. Did Stage 5 engineering completion succeed?
2. What did Stage 5 actually demonstrate?
3. What did Stage 5 not demonstrate?
4. Is the repo ready to enter Stage 6, and why?

==================================================
2. NON-GOALS - DO NOT DRIFT
==================================================

Do NOT do any of the following:

- do not add new training or evaluation code
- do not silently revise main-run metrics
- do not over-claim paper reproduction success
- do not hide that bicubic currently outperforms the model on the reported PSNR/SSIM summaries
- do not convert artifact-first blind eval into a stronger claim than the evidence supports
- do not start Stage 6 experiments here

This issue is successful only if it makes the current Stage 5 state clear,
honest, and useful for the next stage.

==================================================
3. BRANCH EXPECTATION
==================================================

Suggested branch:
- `docs/update`

Stay on a documentation-oriented branch.
Avoid implementation drift.

==================================================
4. REQUIRED INHERITANCE
==================================================

From the current Stage 5 evidence, this issue must inherit and reflect:

1. Stage 5 pipeline is now fully landed:
   - dataset
   - optics configs
   - encoder
   - loss
   - trainer
   - regular eval
   - blind eval hook
2. A real smoke run exists.
3. A real full `phase-only / L=5` main run exists.
4. Matching regular eval and blind eval artifacts exist for that main run.
5. Stage 4 remains the trusted learnability baseline, not something to erase.

Important inheritance detail:

- Stage 5 engineering completion and paper-quality success are not the same claim.
- Stage 6 go/no-go should be based on whether the repo now has a credible,
  reproducible paper-aligned starting point for systematic follow-up work.

==================================================
5. WHAT YOU ARE ALLOWED TO FINALIZE HERE
==================================================

This issue MAY finalize:

1. the Stage 5 final repo-level verdict
2. the explicit Stage 6 go/no-go decision
3. the consolidated wording for what Stage 5 did and did not prove
4. the list of key Stage 5 assumptions and outcomes worth preserving

This issue MUST keep the following explicit and auditable:

1. whether Stage 5 engineering completion passed
2. whether reported model quality reached paper-level success
3. whether current evidence supports entering Stage 6
4. which documents now serve as the final Stage 5 summary layer

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

Prefer updating existing summary-layer docs instead of creating redundant new docs.

At minimum, consider updating:

- `docs/execution/results_summary.md`
- `docs/execution/experiment_log.md`
- `docs/ai/PROJECT_CONTEXT.md`

You may add one concise consolidation doc only if it materially improves clarity,
but prefer not to duplicate information that already belongs in the summary docs.

Important quality requirement:

- Some existing summary docs may be stale or garbled.
- Clean them into readable UTF-8 content if needed.
- After your update, a new reader should be able to understand the current repo
  state without cross-referencing old, stale Stage 4-only summaries.

==================================================
7. REQUIRED DECISION LOGIC
==================================================

Your final documentation must clearly separate at least these three layers:

1. Stage 5 engineering completion:
   - whether the paper-aligned pipeline exists end to end
2. Stage 5 empirical outcome:
   - what the main run and evals actually showed
3. Stage 6 entry decision:
   - whether the repo is ready for systematic follow-up experiments

The docs should explicitly address the likely outcome that:

- Stage 5 engineering completion is `PASS`
- paper-final quality reproduction is `NOT yet demonstrated`
- Stage 6 entry may still be `GO` because the pipeline and evidence base are now real

Do not hide these distinctions.

==================================================
8. REQUIRED ARTIFACT REFERENCES
==================================================

At minimum, the updated docs should reference:

- `docs/execution/stage5_smoke_report.md`
- `docs/execution/stage5_main_run_report.md`
- `outputs/stage5/main_run/stage5_l5_phase_main/run_summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_val_eval/summary.json`
- `outputs/stage5/eval/stage5_l5_phase_main_test_eval/summary.json`
- `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`

If you summarize metrics, keep them faithful to the existing artifact values.

==================================================
9. FILE SCOPE
==================================================

Primary files likely to be changed:

- `docs/execution/results_summary.md`
- `docs/execution/experiment_log.md`
- `docs/ai/PROJECT_CONTEXT.md`

Optional:

- one concise Stage 5 consolidation doc only if strictly justified

Avoid:

- trainer/eval code edits unless a tiny documentation-related correction is unavoidable
- planning-doc rewrites unrelated to final consolidation

==================================================
10. ACCEPTANCE CRITERIA
==================================================

Issue 5.11 is complete only if:

1. The repo clearly states whether Stage 5 passed at the engineering level
2. The repo clearly states what current model quality results do and do not show
3. The Stage 6 go/no-go decision is explicit
4. Summary docs are readable, current, and no longer stuck at Stage 4-only status
5. Key Stage 5 artifact references are preserved

==================================================
11. FINAL RESPONSE FORMAT
==================================================

After finishing, respond with:

- short summary of the consolidation outcome
- exact files changed
- the final Stage 5 engineering verdict
- the final Stage 6 go/no-go decision
- the key empirical caveats that remain true after Stage 5
