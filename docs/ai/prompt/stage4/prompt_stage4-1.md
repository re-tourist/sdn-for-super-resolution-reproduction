You are working in the repository for reproducing the paper
“Super-resolution image display using diffractive decoders”.

Your task is to complete GitHub Issue 4.1:

Issue title:
Freeze Stage-4 minimal protocol and tensor contract

You must create or update the document:

docs/plan/stage4_protocol_freeze.md

The goal of this issue is NOT to implement code.
The goal is to write a high-quality engineering protocol document that freezes the Stage 4 minimal closed-loop scope and tensor contract, so later coding work can proceed without ambiguity.

## Critical project context

This repo is already past:
- Stage 0 repo scaffold and planning
- Stage 1 / 2 baseline and evaluation groundwork
- Stage 3 optical module verification

The repo is now entering Stage 4.

Stage 4 has exactly one core question:

Can the full system learn under gradient-driven end-to-end training after connecting an encoder to the existing optical decoder?

Do NOT expand this into:
- Stage 5 paper-aligned full settings
- Stage 6 systematic ablations
- hardware robustness / quantization / misalignment / blind line-pair evaluation

Those are explicitly out of scope for this issue.

## Source-of-truth priority

When documents disagree, prioritize in this order:
1. docs/ai/PROJECT_CONTEXT.md
2. docs/plan/plan_overview.md
3. docs/plan/stage3_contract_freeze.md
4. current code in src/ and scripts/
5. docs/paper/paper_notes.md
6. docs/execution/results_summary.md
7. docs/execution/experiment_log.md

Do not invent facts that are not supported by these sources.
If something cannot be confirmed from the repo, label it explicitly as:
- 假设
- 待确认

## Facts that must be respected

1. The optical contract is already frozen in Stage 3:
   phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi

2. Stage 4 must build on the existing optical core, not redesign it.

3. The current diffractive decoder already exposes Stage-4-friendly hooks such as:
   - forward_from_phase(...)
   - forward_from_field(...)
   - forward_from_phase_provider(...)

4. The current Stage 3 training/verification scripts are still toy/synthetic verification scripts.
   In particular, the decoder-only small-subset script uses:
   - synthetic target patterns
   - toy grid sizes such as 24 / 32 / 48 / 20
   This is useful for optical learnability sanity, but is NOT yet a real Stage 4 end-to-end training pipeline.

5. The repo does NOT yet have a real Stage 4 general trainer / eval runner / config path.
   Be honest about this gap.

6. The trusted training skeleton currently available is the electronic baseline training path.
   Reuse that as a reference anchor conceptually, but do not pretend Stage 4 trainer already exists.

## Files you must inspect before writing

Read these files carefully and ground the document in them:

- docs/ai/PROJECT_CONTEXT.md
- docs/plan/plan_overview.md
- docs/plan/stage3_contract_freeze.md
- docs/paper/paper_notes.md
- src/models/optics/diffractive_decoder.py
- scripts/train_electronic_baseline.py
- scripts/train_decoder_only_small_subset.py

If additional nearby files are needed to clarify naming or repo conventions, inspect them too, but keep the task focused.

## What the document must achieve

The document must freeze the first Stage 4 protocol clearly enough that later implementation issues can build on it.

It must answer, in concrete engineering terms:

1. What Stage 4 is trying to prove
2. What Stage 4 is explicitly NOT trying to do
3. What exact tensor/dataflow boundary is inherited from Stage 3
4. What parts of the current repo are reusable
5. What infrastructure is still missing
6. What the first minimal closed-loop training setup is
7. What the first loss/readout choice is
8. What the first acceptance checks are
9. What evidence is required before entering Stage 5

## Required document structure

Write the document in Chinese, professional and concise, with clear engineering tone.

It must contain these sections:

1. 文档目的
2. Stage 4 核心问题
3. Stage 4 范围边界
4. 当前仓库状态摘要
5. 已冻结的 Stage 3 optical contract
6. Stage 4 最小闭环定义
7. 首版张量契约（tensor contract）
8. 首版训练协议
9. 首版损失与读出约定
10. 验收标准
11. 非目标与延后事项
12. 对后续 issue 的约束

## Content requirements

### In “当前仓库状态摘要”, explicitly distinguish:
- 已可复用资产
- 当前仍缺失的基础设施
- 当前哪些验证是可信的
- 当前哪些还只是初步 toy / synthetic verification

### In “Stage 4 最小闭环定义”, make the minimal system explicit:
HR input -> minimal encoder -> phi_lr -> optical decoder -> I_out_roi -> loss

### In “首版张量契约（tensor contract）”, define at minimum:
- upstream input meaning
- encoder output meaning
- phi_lr tensor role
- optical decoder input/output responsibilities
- expected shape semantics
- phase-domain responsibility boundary
- what is fixed vs what remains configurable

If exact sizes cannot be safely frozen yet, state that clearly as 待确认 and separate:
- already frozen contract
- deferred implementation choice

### In “首版训练协议”, explicitly freeze the first acceptance protocol:
- single-sample overfit
- small-subset closed-loop training
- artifact saving expectations
- gradient observability expectations

### In “首版损失与读出约定”, make a pragmatic Stage 4 choice.
You must avoid pretending that paper-final loss/alignment is already required.
Be honest if the first Stage 4 loss/readout choice is intentionally simplified.

### In “非目标与延后事项”, explicitly push these out of Stage 4:
- paper-exact full setting alignment
- line-pair blind test
- quantization sweep
- misalignment robustness
- L=1/3/5 systematic comparison
- efficiency penalty ablation unless needed later
- large-scale training

### In “对后续 issue 的约束”, write the document so that future issues such as:
- minimal encoder
- hybrid wrapper
- Stage 4 trainer
- single-sample overfit
- small-subset run
must all obey this protocol.

## Output style requirements

- Chinese only
- Use Markdown
- Do not write chatty explanations
- Do not write a generic tutorial
- Do not rewrite Stage 5 planning into this document
- Do not claim uncertain facts as certain
- Be explicit, engineering-oriented, and repo-grounded

## Implementation instructions

1. Inspect the required files first.
2. Draft the document content.
3. Create or update docs/plan/stage4_protocol_freeze.md.
4. Ensure the final document is internally consistent and clearly separates:
   - frozen facts
   - chosen Stage 4 simplifications
   - deferred decisions
5. If needed, make only minimal wording-level edits to keep terminology consistent with existing docs.
6. Do not modify unrelated code or documents.

## Final response format

After editing the file, respond with:
- a short summary of what was written
- the exact file path changed
- a concise bullet list of the key frozen decisions
- any explicit assumptions / pending confirmations that remain

Important:
This document is a protocol-freeze document, not a brainstorming note.
It must make later implementation ambiguity smaller, not larger.
If a statement does not constrain later implementation decisions, do not include it.