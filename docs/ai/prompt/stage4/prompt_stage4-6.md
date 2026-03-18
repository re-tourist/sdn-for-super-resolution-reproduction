You are working on Stage 4 of the repo. Execute Issue 4.6 only.

Goal:
Use the existing Stage 4 minimal closed-loop trainer to run the formal single-sample overfit acceptance experiment and write an honest acceptance report.

Critical boundaries:
- DO NOT rewrite the optical core.
- DO NOT rewrite dataset / adapter / wrapper.
- DO NOT expand into Stage 5 or a general training framework.
- DO NOT redesign the Stage 4 trainer unless a tiny bugfix is strictly required to complete the acceptance run.
- Reuse the existing script: scripts/train_stage4_minimal.py
- This issue is an execution + inspection + reporting issue, not a new architecture issue.

Context from Issue 4.5:
- The minimal trainer already exists and runs.
- It supports --single-sample and --subset-size.
- It already saves history, summaries, previews, checkpoints, loss curves, and grad stats.
- CPU smoke already showed:
  - single-sample loss decreased slightly
  - encoder gradients observed
  - optics gradients observed
- Therefore, for Issue 4.6, the task is to run a stronger single-sample acceptance experiment and document the result honestly.

What to do:
1. Inspect scripts/train_stage4_minimal.py and confirm the proper CLI usage for a formal single-sample run.
2. Run a formal single-sample overfit experiment with a stronger step count than smoke.
   - Start with a reasonable acceptance setting such as:
     python scripts/train_stage4_minimal.py --single-sample --steps 100
   - If the result is still ambiguous and the script is stable, extend to a stronger run such as 300 or 500 steps.
   - Prefer GPU if available; otherwise run on CPU and state the limitation honestly.
3. Collect and inspect all artifacts:
   - config_snapshot.json
   - history.json
   - run_summary.json
   - loss_curve.png
   - preview_step0.png
   - preview_best.png
   - preview_final.png
   - phi previews
   - checkpoints
   - grad_stats.json
4. Evaluate acceptance against the issue intent:
   - Does loss decrease clearly?
   - Does output ROI move closer to target?
   - Are encoder and optical gradients non-zero?
   - Are there any NaN/Inf or numerical issues?
5. Write a concise but honest report at:
   docs/execution/stage4_single_sample_report.md

Report requirements:
- State exact command(s) used
- State device used
- State output artifact directory
- Summarize initial loss, best loss, final loss
- Summarize whether encoder and optics gradients were observed
- Summarize preview observations
- Give a clear PASS / FAIL / BORDERLINE verdict
- If FAIL or BORDERLINE, explain the most likely reason without overclaiming
- Include a short “next step” section:
  - if pass: recommend proceeding to the next minimal Stage 4 acceptance task
  - if fail: recommend the narrowest next debugging target, not a broad redesign

Implementation style:
- Minimal changes only.
- If a tiny bugfix is necessary to complete the acceptance run, keep it surgical and explain it clearly.
- Do not hide failure.
- Do not artificially relax acceptance wording to force a pass.

Deliverables:
- completed acceptance run artifacts
- docs/execution/stage4_single_sample_report.md

At the end, provide:
1. a short summary of what was run,
2. verdict (PASS / FAIL / BORDERLINE),
3. files changed,
4. exact next recommendation.