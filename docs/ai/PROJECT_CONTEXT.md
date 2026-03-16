# PROJECT_CONTEXT

## 1. Project Overview

**Project Name:**  
Optical Neural Network Super-Resolution Reproduction

**Primary Goal:**  
Reproduce the paper *Super-resolution image display using diffractive decoders* with a traceable engineering workflow.

**Current Stage:**  
The project is entering **Stage 4: minimal closed-loop training**.

**Important Stage-4 Principle:**  
The goal is **not** to chase the paper's final metric yet.  
The goal is to answer one question:

> Can the full system learn under gradient-driven training once the encoder is connected to the optical decoder?

---

## 2. Stage Status Snapshot

### Completed / trusted

- Stage 0 repo scaffold and planning docs
- Stage 1 / 2 data sanity path on EMNIST
- interpolation baseline
- pure electronic bottleneck baseline
- unified PSNR / SSIM evaluation helpers
- run-order documentation
- experiment logging / troubleshooting / results summary
- Stage 3 optical decoder skeleton
- Stage 3 optical forward sanity checks
- Stage 3 decoder-only single-sample fitting
- Stage 3 decoder-only small-subset capacity checks

### Current focus

- build the first **encoder + optical decoder** minimal closed loop
- keep the setup small, debuggable, and explicitly non-paper-final
- verify learnability before any Stage 5 paper-alignment work

---

## 3. What Stage 4 Means In This Repo

Stage 4 in this repo means:

1. connect a minimal encoder
2. feed encoder output into the existing optical contract
3. train on a very small dataset first
4. run single-sample overfit and small-subset sanity checks
5. save enough intermediate artifacts to debug gradients, phase output, and optical readout

Stage 4 does **not** mean:

- full paper hyperparameter alignment
- full `96 -> learned phase pattern -> paper-exact optics` reproduction
- quantization, misalignment, or robustness studies
- large-scale sweeps or ablations

Those belong to later stages.

---

## 4. Hard Constraints A New Assistant Must Know

### 4.1 Optical contract is already frozen

The Stage 3 optical path is already defined as:

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`

Any Stage 4 plan must build on that contract rather than redesigning the optical core.

### 4.2 The optical core already exposes Stage-4 hooks

The current optical decoder already supports:

- `forward_from_phase(...)`
- `forward_from_field(...)`
- `forward_from_phase_provider(...)`

This means Stage 4 should primarily solve the **upstream encoder / trainer / protocol** problem, not rewrite optics first.

### 4.3 Electronic baseline is already trusted

The pure electronic bottleneck baseline has passed:

- single-sample overfit sanity
- small-subset training sanity

So Stage 4 should reuse it as a training/reference anchor instead of re-opening Stage 1 / 2 questions unless new evidence appears.

### 4.4 Current Stage 3 training scripts are still toy verification scripts

The Stage 3 decoder-only scripts currently use:

- synthetic target patterns
- toy grid sizes such as `24 / 32 / 48 / 20`

They are useful for optical learnability verification, but they are **not yet** the Stage 4 end-to-end data pipeline.

### 4.5 Some infrastructure is still missing

The repo does **not yet** have:

- a real Stage 4 general trainer
- a real Stage 4 eval runner
- Stage 4 optics configs
- a clean dataset module under `src/datasets/`
- a `tests/` directory for systematic automated checks

At the time of this update:

- `scripts/train.py` is empty
- `scripts/eval.py` is empty
- `scripts/visualize.py` is empty

So any Stage 4 plan must honestly treat these as pending work.

---

## 5. Current Code Assets That Matter Most

### Optical path

- `src/models/optics/diffractive_decoder.py`
- `src/models/optics/phase_provider.py`
- `src/models/optics/propagation.py`
- `src/models/optics/readout.py`
- `src/models/optics/phase_utils.py`

### Stage 3 verification scripts

- `scripts/check_optical_readout.py`
- `scripts/check_optical_forward_depths.py`
- `scripts/train_decoder_only_single_sample.py`
- `scripts/train_decoder_only_small_subset.py`
- `scripts/train_decoder_only_small_subset_sweep.py`

### Stage 1 / 2 trusted baseline path

- `scripts/train_electronic_baseline.py`
- `src/models/electronic_baseline.py`
- `src/eval/evaluator.py`
- `src/eval/metrics.py`

---

## 6. Current Documentation Sources Of Truth

When different documents disagree, a new assistant should prioritize them in this order:

1. `docs/plan/plan_overview.md`
2. `docs/plan/stage3_contract_freeze.md`
3. `docs/integration/stage3_unresolved_params.md`
4. current code in `src/` and `scripts/`
5. `docs/paper/paper_notes.md`
6. `docs/execution/results_summary.md`
7. `docs/execution/experiment_log.md`

This ordering exists because some older background/context docs may lag behind the repo's actual progress.

---

## 7. Immediate Stage-4 Engineering Gaps

The most likely Stage 4 work items are:

1. define the minimal encoder output contract
2. connect encoder output to `phi_lr`
3. decide the first Stage 4 training scale
   - stay toy first
   - or move partially toward the paper setup
4. build a real Stage 4 trainer script
5. build Stage 4 config files
6. define artifact saving rules
7. define Stage 4 acceptance checks
   - loss decreases
   - single-sample overfit works
   - gradients reach encoder and optics
   - intermediate outputs are interpretable

---

## 8. Known Risks

The main current risks are:

1. confusing Stage 3 toy verification with Stage 4 real closed-loop training
2. jumping straight to paper-aligned full settings before proving end-to-end learnability
3. hiding unresolved choices such as phase range mapping or training scale selection
4. writing a large trainer before defining a small acceptance protocol
5. losing traceability by not saving configs, summaries, and intermediate visualizations

---

## 9. Recommended Minimal File Package For A New Chat

If a user wants another assistant to plan Stage 4, the minimal useful package is:

- `docs/ai/PROJECT_CONTEXT.md`
- `docs/plan/plan_overview.md`
- `docs/plan/stage3_contract_freeze.md`
- `docs/paper/paper_notes.md`
- `src/models/optics/diffractive_decoder.py`
- `scripts/train_electronic_baseline.py`
- `scripts/train_decoder_only_small_subset.py`
- `docs/execution/results_summary.md`

If the user can provide a bit more context, add:

- `docs/integration/stage3_unresolved_params.md`
- `docs/integration/sdn_optics_contract.md`
- `docs/execution/experiment_log.md`
- `outputs/optics/decoder_only_small_subset_sweep_run1/depth_comparison.csv`

---

## 10. One-Screen Summary

This is a long-running reproduction project for a hybrid
`encoder -> diffractive optical decoder` super-resolution paper.

The repo has already passed:

- Stage 1 / 2 baseline and evaluation groundwork
- Stage 3 optical module verification

The repo is now entering Stage 4:

- connect a minimal encoder
- reuse the frozen optical contract
- run the first end-to-end minimal closed loop
- prove the system can learn

The biggest missing piece is **not optics core**, but the **Stage 4 training pipeline around it**.
