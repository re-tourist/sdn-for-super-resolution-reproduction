# PROJECT_CONTEXT

## 1. Project Overview

**Project Name:**  
Optical Neural Network Super-Resolution Reproduction

**Primary Goal:**  
Reproduce the paper *Super-resolution image display using a diffractive optical network* with a high-quality engineering workflow, starting from the numerical phase-only mainline, then extending to ablations such as complex-valued encoding, quantization, and misalignment robustness.

**Current Intent:**  
The current goal is not to build a fully novel system yet, but to complete a rigorous reproduction pipeline that is:
- technically correct,
- experimentally traceable,
- easy to extend later into optical reconstruction or hybrid optical-electronic research.

---

## 2. Target Paper

**Paper Title:**  
Super-resolution image display using a diffractive optical network

**High-Level Understanding:**  
The paper proposes a jointly trained hybrid system:
- a **digital/electronic encoder** generates a **low-resolution modulation pattern**,
- an **all-optical diffractive decoder** reconstructs a **higher-resolution output image** in the output field of view.

**Important Conceptual Points:**
- The low-resolution input is **not just a normal low-res image**, but a learned **optical-friendly encoding / modulation pattern**.
- The optical decoder is the main all-optical reconstruction / synthesis module.
- The paper studies:
  - phase-only vs complex-valued encoding
  - 1 / 3 / 5 diffractive layers
  - quantization robustness
  - misalignment vaccination / robustness
  - output FOV efficiency

**Current Reproduction Focus:**  
Start from the **numerical phase-only mainline**, not the full experimental THz hardware pipeline.

---

## 3. Project Scope

### In Scope (current)
- numerical reproduction of the main optical pipeline
- dataset preparation and sanity checks
- interpolation baseline
- pure electronic baseline
- optical propagation module
- optical decoder capacity tests
- encoder + optical decoder end-to-end pipeline
- paper-aligned experiments and ablations

### Out of Scope (for now)
- full real hardware reproduction
- full THz experimental setup
- ambitious new optical architecture redesign
- large-scale world-model integration
- advanced quantization-aware training as the first milestone

---

## 4. Current Development Strategy

The development is intentionally staged.

### Stage philosophy
The project is not implemented end-to-end all at once.  
Instead, it is built in the following order:

1. dataset sanity check
2. evaluation pipeline
3. interpolation baseline
4. pure electronic baseline
5. optical propagation verification
6. optical decoder capacity verification
7. encoder + optical decoder minimal closed loop
8. paper-aligned experiments
9. ablations
10. robustness tests

### Important Principle
Before using the optical model, the following must already be trustworthy:
- data pipeline
- HR/LR construction
- interpolation baseline
- PSNR/SSIM pipeline
- experiment logging

---

## 5. Current Project Status

### Completed
- Stage 0 initialization has been done.
- Basic repo scaffolding and some planning documents already exist.
- `overview.md` has been patched to better match the current paper.
- `paper_notes.md` and `checklist.md` have been introduced / expanded.
- Step 0 planning for Stage 1 and Stage 2 has already been clarified.
- Issue 1 (dataset inspection script) has been implemented by Codex and is currently in **manual verification pending / provisional pass** state.

### Current Focus
We are currently in:
- **Stage 1 / Stage 2 groundwork**
- specifically around:
  - dataset sanity check
  - evaluation pipeline planning
  - baseline preparation

### Immediate Next Step
- manually verify the dataset inspection script output
- then move to:
  - unified evaluation pipeline
  - interpolation baseline

---

## 6. Current Understanding of the Data Pipeline

### Raw Data Source
The current raw source is **EMNIST**, not MNIST.

### Important Clarification
At the current Stage 1 / 2:

- EMNIST provides the **raw source images**
- the project currently treats these as the source for **HR images**
- **LR images are artificially constructed** by downsampling HR
- upsampled LR is used to inspect interpolation behavior and baseline quality

So at this stage, the current task shell is:

`HR -> downsample -> LR -> upsample / baseline -> reconstructed HR`

This is a **sanity-check SR shell**, not yet the final learned optical pattern pipeline.

### Important Distinction
This means:
- current dataset inspection is checking the **basic HR/LR pipeline**
- it is **not yet identical to the final learned low-resolution optical modulation pattern** used in the full paper pipeline

### Current Open Practical Choice
The project may use one of two HR definitions depending on the implementation phase:
1. a simpler sanity-check HR definition
2. a more paper-aligned larger HR image (e.g. 96×96 setup)

This must be explicitly recorded in experiment logs whenever changed.

---

## 7. Branching Strategy

### Main branches
- `main` → stable milestone versions
- `dev` → current integration branch

### Other branches
All non-main branches should be created **from `dev`**, unless a true release hotfix from `main` is needed.

### Long-lived category branches
The current preferred style is **category-based**, not overly stage-specific:

- `feat/data`
- `feat/scripts`
- `feat/baseline`
- `feat/model`
- `feat/train`
- `feat/eval`
- `feat/test`
- `feat/optics`
- `docs/update`
- `hotfix/<name>`

### Rationale
We intentionally avoid creating too many ultra-specific stage branches, because that causes:
- low branch reuse
- branch explosion
- poor long-term maintainability

Instead:
- branches provide **category-level organization**
- issues / commits / logs provide **fine-grained traceability**

---

## 8. Documentation Structure

The docs are organized by function, not by chat history.

### Recommended structure

- `docs/plan/`
  - long-term plans
  - execution plans
- `docs/paper/`
  - paper notes
  - PDFs
- `docs/execution/`
  - experiment logs
  - troubleshooting
  - results summary
  - checklist
- `docs/ai/`
  - prompts
  - Codex / agent feedback
  - run order
- `docs/gitflow/`
  - git workflow notes

### Important Principle
Chat history is not the project memory system.  
The repo docs are the project memory system.

---

## 9. Important Existing / Expected Documents

### Planning
- `docs/plan/overview.md`
- `docs/plan/reproduction_plan.md`
- `docs/plan/execution_round_1_2.md`

### Paper
- `docs/paper/paper_notes.md`

### Execution
- `docs/execution/checklist.md`
- `docs/execution/experiment_log.md`
- `docs/execution/results_summary.md`
- `docs/execution/troubleshooting.md`

### AI workflow
- `docs/ai/prompts/...`
- `docs/ai/codex_feedback/...`
- `docs/ai/run_order.md`

### Context
- `docs/PROJECT_CONTEXT.md` (this file)

---

## 10. Current Stage 1 / Stage 2 Task Breakdown

### Stage 1
Build a trustworthy baseline reference.

Current intended subtasks:
- dataset inspection
- interpolation baseline
- minimal pure electronic baseline
- single-sample overfitting
- small-subset training

### Stage 2
Stabilize the data and evaluation pipeline.

Current intended subtasks:
- unified PSNR/SSIM implementation
- sample visualization standardization
- output metric export
- evaluation protocol documentation

### Important Ordering Constraint
The preferred execution order is:

1. dataset inspection
2. unified evaluation script
3. interpolation baseline
4. pure electronic baseline
5. single-sample overfit
6. small-subset training

---

## 11. Current AI-Assisted Development Workflow

This project uses AI coding assistance, so tasks are intentionally designed to be:
- narrow in scope
- easy to verify
- low in cross-file coupling
- issue-oriented

### Preferred task style
Each AI task should have:
- a clear issue
- a branch category
- a small and local modification range
- explicit acceptance criteria
- explicit "do not touch" boundaries

### Important Workflow Principle
For AI-generated code:
- merge to `dev` first
- never merge directly to `main`
- always record the task in logs / feedback docs

---

## 12. Current Known Open Questions

These are not fully resolved yet and may reappear in later conversations:

1. what exact HR definition should be used in the earliest sanity-check SR shell
2. how closely the early dataset should follow the paper’s larger-image setup
3. when to introduce paper-aligned mosaic / larger image construction
4. when to switch from simple HR/LR shell to learned low-resolution optical pattern pipeline
5. whether phase-only should remain the sole mainline for the first full reproduction milestone
6. when to add efficiency penalty
7. when to introduce quantization evaluation and misalignment robustness tests

---

## 13. Reproduction Priorities

The current priority order is:

1. correctness
2. traceability
3. reproducibility
4. paper alignment
5. speed

This means:
- we do not skip sanity checks just to reach the optical model faster
- we do not trust a result that cannot be traced to config / commit / issue / log

---

## 14. What a New Conversation Should Know Immediately

If this file is pasted into a new conversation, the new assistant should understand:

- this is a long-running optical neural network reproduction project
- the target paper is about digital encoder + diffractive optical decoder for super-resolution image display
- the project is currently still in the Stage 1 / 2 groundwork phase
- Stage 0 planning is already done
- dataset inspection has already been implemented and awaits manual verification
- the next likely steps are:
  - evaluate the inspection output
  - build the unified metric pipeline
  - run interpolation baseline
- the project uses category-based git branches from `dev`
- documentation is treated as the long-term memory of the project

---

## 15. How to Use This File

When starting a new conversation, paste this file first and say:

> “This is my project context. Please read it before answering. Continue from the current project state.”

If needed, also paste one or two of:
- `docs/paper/paper_notes.md`
- `docs/execution/experiment_log.md`
- `docs/plan/overview.md`

depending on whether the new conversation is about:
- paper understanding
- engineering execution
- debugging
- experiment interpretation

---

## 16. Maintenance Rule

This file should be updated only when one of the following changes:

- project stage changes
- reproduction scope changes
- branch strategy changes
- documentation structure changes
- current next-step focus changes
- major architectural decision changes

It should **not** be updated for every single experiment result.  
Detailed experiment information belongs in:
- experiment logs
- result summaries
- troubleshooting notes

---

## 17. Current Minimal Quick Summary

**Paper:** diffractive optical decoder for super-resolution display  
**Current phase:** Stage 1 / 2 groundwork  
**Done:** stage planning, docs patching, dataset inspection script implementation  
**Pending:** manual inspection verification, eval pipeline, interpolation baseline  
**Workflow:** category branches from `dev`, docs as long-term project memory  
**Immediate next step:** verify dataset inspection output and then implement unified evaluation