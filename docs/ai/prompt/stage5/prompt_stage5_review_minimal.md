You are assisting a research-engineering repository for reproducing
"Super-resolution image display using diffractive decoders".

Your task is not to write code. Your task is to review the current Stage 5 planning
documents so that a later Codex prompt can be sharper, narrower, and more executable.

==================================================
0. READ ONLY THESE 5 FILES FIRST
==================================================

Before doing anything, read these files in order:

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/paper/paper_notes.md`
4. `docs/plan/stage_plan/stage5/stage5_plan.md`
5. `docs/plan/stage_plan/stage5/stage5_issue_plan.md`

Do not assume any extra context by default.

If you think some claim cannot be verified from these 5 files alone, say:
- `needs extra evidence`

Do not invent missing facts.

==================================================
1. REVIEW GOAL
==================================================

Review these two target documents:

- `docs/plan/stage_plan/stage5/stage5_plan.md`
- `docs/plan/stage_plan/stage5/stage5_issue_plan.md`

Your goal is to evaluate whether they are good enough to support future
high-quality Codex implementation prompts for Stage 5 work.

This is an engineering-review task plus prompt-quality-review task.
It is not a writing-style review.

==================================================
2. WHAT TO CHECK
==================================================

Focus on these questions:

1. Is the Stage 5 boundary clear?
2. Does Stage 5 stay aligned with the paper rather than drifting into Stage 6?
3. Does it correctly inherit the frozen optical contract and Stage 4 learnability baseline?
4. Are the issue breakdown and dependency order executable?
5. Are Goal / Tasks / Deliverable / Acceptance specific enough for future Codex prompts?
6. Are important assumptions and unresolved items recorded explicitly?
7. Are the suggested branches coherent with the repo's branch conventions?
8. What prompt-critical context is still missing if we later want Codex to implement one Stage 5 issue?

==================================================
3. IMPORTANT RULES
==================================================

- Do not ask for extra documents unless they are truly required to verify a specific claim.
- Do not restate the entire documents.
- Do not rewrite Stage 5 from scratch.
- Do not treat Stage 6 work as a Stage 5 requirement.
- Do not present paper details as certain if the documents still mark them as unresolved.
- If a point is acceptable but still slightly risky for future prompting, call it out clearly.

==================================================
4. REQUIRED OUTPUT FORMAT
==================================================

Use exactly this structure:

A. Findings
- Put problems first
- Order by severity: High / Medium / Low
- Cite the target document when possible
- Focus on issues that would make future Codex prompts drift, overreach, or become underspecified
- If no serious issues exist, explicitly say `No critical findings`

B. Strengths
- State what these Stage 5 docs already do well
- Explain why those strengths help future Codex prompts

C. Gaps For Prompting Codex
- Split into:
  - `Must add before implementation prompts`
  - `Helpful but optional`
- Only include context gaps that matter for implementable prompts

D. Prompt-Readiness Verdict
- Choose one:
  - `Ready`
  - `Ready with gaps`
  - `Not ready`
- Explain the decision in 3 to 6 sentences

E. Next Prompt Package
- List the smallest file bundle you would attach to the next Codex implementation prompt
- If one extra file is needed beyond the original 5, explain why

F. One Better Meta-Prompt
- Describe the minimum fields that any future Stage 5 implementation prompt to Codex should contain
- Keep it practical and implementation-oriented

==================================================
5. REVIEW STYLE
==================================================

- Be direct and engineering-oriented
- Prefer concrete criticism over vague praise
- Keep the review concise but specific
- If a claim needs extra evidence, say so plainly
