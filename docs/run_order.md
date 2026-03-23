## Environment Setup

Linux server minimal setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the server uses NVIDIA GPU and needs a specific CUDA build of PyTorch, install matching
`torch` / `torchvision` first, then run:

```bash
pip install -r requirements.txt
```

## Data Download Behavior

This project currently uses `torchvision.datasets.EMNIST`.

- If you run `scripts/train_electronic_baseline.py` or `scripts/run_interpolation_baseline.py` and the dataset is missing, they will **not** auto-download by default, because `--download` defaults to `False`.
- On a fresh server that only has code and no data, you should add `--download` the first time you run those scripts.
- `scripts/inspect_dataset.py` is different: its `--download` default is `True`, so it will try to download EMNIST automatically if data is absent.
- Default dataset root fallback is `data/raw/emnist`. You can also pass `--dataset-root <your_emnist_root>`.

Recommended first-time data preparation on a new server:

```bash
python scripts/inspect_dataset.py \
  --dataset-root data/raw/emnist \
  --download \
  --num-samples 4 \
  --output-dir outputs/inspection/bootstrap_check
```

## Run Order

### 1) Interpolation baseline (bicubic)

```bash
python scripts/run_interpolation_baseline.py \
  --split train \
  --num-samples 64 \
  --seed 42 \
  --output-dir outputs/interpolation/train \
  --mode bicubic \
  --download
```

### 2) Electronic baseline: overfit 1 sample

Explanation: use `--overfit-single-sample` so train/val share the exact same sample.
This checks whether the bottleneck autoencoder can memorize a single HR target under the
same low-resolution latent constraint. Do not use `num-samples=2 + val-ratio=0.5` for this,
because that creates two different samples instead of a true overfit sanity check.

```bash
python scripts/train_electronic_baseline.py \
  --output-dir outputs/electronic_baseline/fit_one_sample \
  --epochs 200 \
  --batch-size 1 \
  --lr 1e-3 \
  --seed 42 \
  --hr-size 96 \
  --sr-factor 4 \
  --latent-channels 1 \
  --num-samples 1 \
  --num-workers 0 \
  --download \
  --save-samples \
  --overfit-single-sample
```

### 3) Electronic baseline: small dataset training

```bash
python scripts/train_electronic_baseline.py \
  --output-dir outputs/electronic_baseline/smallset_e10 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42 \
  --hr-size 96 \
  --sr-factor 4 \
  --latent-channels 1 \
  --num-samples 2000 \
  --val-ratio 0.1 \
  --num-workers 4 \
  --download \
  --save-samples
```

If data has already been downloaded, you can change `--download` to `--no-download` and
optionally add `--dataset-root <your_emnist_root>`.

### 4) Stage 4 minimal closed-loop: formal single-sample acceptance

Issue 4.6 is an execution/reporting task, not a new trainer design task. Use the existing
`scripts/train_stage4_minimal.py` as-is and run the formal single-sample overfit acceptance
experiment on the Linux server. Start with 100 steps. Only extend to a stronger run if the
100-step result is still ambiguous.

Recommended first acceptance run:

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 100 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --download \
  --run-name issue4_6_single_sample_100
```

If the 100-step result is still borderline but numerically stable, extend to 300 steps:

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 300 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_6_single_sample_300
```

After the run finishes, collect at least these files from the output directory:

- `config_snapshot.json`
- `history.json`
- `run_summary.json`
- `loss_curve.png`
- `preview_step0.png`
- `preview_best.png`
- `preview_final.png`
- `phi_preview_step0.png`
- `phi_preview_best.png`
- `phi_preview_final.png`
- `grad_stats.json`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_latest.pt`

Expected output roots:

- `outputs/stage4/minimal_trainer/issue4_6_single_sample_100/`
- `outputs/stage4/minimal_trainer/issue4_6_single_sample_300/`

### 5) Stage 4 minimal closed-loop: formal small-subset acceptance

Issue 4.7 starts only after Issue 4.6 has already passed. Reuse the same
`scripts/train_stage4_minimal.py` stack and do not redesign the trainer,
optics, dataset path, or wrapper. The goal is to check whether the learnability
seen in single-sample overfit extends to a small real subset without obvious
collapse.

Recommended first acceptance run:

```bash
python scripts/train_stage4_minimal.py \
  --subset-size 16 \
  --batch-size 4 \
  --steps 200 \
  --preview-limit 4 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --download \
  --run-name issue4_7_small_subset_16_s200
```

If the 200-step result is still ambiguous but numerically stable, extend to 400
steps with the same subset size:

```bash
python scripts/train_stage4_minimal.py \
  --subset-size 16 \
  --batch-size 4 \
  --steps 400 \
  --preview-limit 4 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_7_small_subset_16_s400
```

After the run finishes, collect at least these files from the output directory:

- `config_snapshot.json`
- `history.json`
- `run_summary.json`
- `loss_curve.png`
- `preview_step0.png`
- `preview_best.png`
- `preview_final.png`
- `phi_preview_step0.png`
- `phi_preview_best.png`
- `phi_preview_final.png`
- `grad_stats.json`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_latest.pt`

For Issue 4.7 specifically, the previews must be kept because they are the main
evidence for:

- multiple different inputs
- corresponding different outputs
- whether there is obvious collapse to one generic pattern

Expected output roots:

- `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s200/`
- `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s400/`

## Notes

- `scripts/train_electronic_baseline.py` now treats explicit CLI arguments as higher priority
  than `configs/base.yaml`. If you pass `--epochs 10 --batch-size 32 --lr 1e-3`, those values
  will be used directly and will not be overwritten by config defaults.
- `src/models/electronic_baseline.py` no longer uses the old `ReLU + sigmoid` output path
  that collapsed to near-black reconstructions on sparse EMNIST images. The current model uses
  `LeakyReLU` in hidden layers and a bounded `atan` mapping at the output.
- The `samples/epoch_XXX.png` files now show multiple fixed validation samples in one grid.
  Earlier versions only saved the first validation sample each epoch, which made the outputs
  look like the same image was being repeated.

### 6) Stage 5 main run prep: phase-only L=5 on Linux server

This section prepares the real Stage 5 main run for Linux-server execution.
Do **not** treat the local smoke run as the main result, and do **not** launch
this full long run in the local workspace.

Dedicated config:
- `configs/stage5/stage5_trainer_main.yaml`

Recommended shell variables on the Linux server:

```bash
RUN_NAME=stage5_l5_phase_main
TRAIN_ROOT=outputs/stage5/main_run/${RUN_NAME}
BEST_CKPT=${TRAIN_ROOT}/checkpoints/checkpoint_best.pt
LATEST_CKPT=${TRAIN_ROOT}/checkpoints/checkpoint_latest.pt
```

First-run notes:
- the config targets the full Stage 5 main budget: `L=5`, `batch_size=40`, `steps=750000`, `validate_every=1500`
- the runtime also enables `log_every=25`, `checkpoint_every=100`, and `keep_history_in_memory=false` for long-run observability and bounded checkpoint size
- the default dataset root is `data/raw/emnist`
- if EMNIST is not already present on the server, add `--download` to the **first** training command only
- if the server only has one visible GPU, you can ignore `CUDA_VISIBLE_DEVICES`
- if the server has multiple GPUs and you want to pin one GPU explicitly, prefix any command with `CUDA_VISIBLE_DEVICES=<id>`

Simplest main training command:

```bash
python scripts/train_stage5_paper.py \
  --config configs/stage5/stage5_trainer_main.yaml \
  --run-name ${RUN_NAME} \
  --device cuda \
  --download
```

If the dataset has already been downloaded, use the same command without `--download`.

Optional single-GPU pinning form:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_stage5_paper.py \
  --config configs/stage5/stage5_trainer_main.yaml \
  --run-name ${RUN_NAME} \
  --device cuda \
  --download
```

Operational monitoring during training:
- `tail -f ${TRAIN_ROOT}/train.log`
- `cat ${TRAIN_ROOT}/status.json`
- `ls ${TRAIN_ROOT}/checkpoints`
- `nvidia-smi`

Incremental files that now update during the run:
- `${TRAIN_ROOT}/train.log`
- `${TRAIN_ROOT}/status.json`
- `${TRAIN_ROOT}/history.jsonl`
- `${TRAIN_ROOT}/grad_stats.jsonl`
- `${TRAIN_ROOT}/loss_curve_points.json`

Important runtime notes:
- first-time EMNIST download/extract can still spend time before the first training step; this now appears in `train.log`, but `torchvision` itself does not give a rich progress bar
- `checkpoint_latest.pt` is updated every `100` steps and at validation points, not every step
- the main config keeps long per-step history out of in-memory checkpoint payloads; use `history.jsonl` and `grad_stats.jsonl` as the authoritative full run trace during long execution

Resume command:

```bash
python scripts/train_stage5_paper.py \
  --config configs/stage5/stage5_trainer_main.yaml \
  --device cuda \
  --resume ${LATEST_CKPT}
```

Regular eval on val:

```bash
python scripts/eval_stage5_paper.py \
  --config configs/stage5/stage5_eval.yaml \
  --checkpoint ${BEST_CKPT} \
  --split val \
  --run-name ${RUN_NAME}_val_eval \
  --device cuda
```

Regular eval on test:

```bash
python scripts/eval_stage5_paper.py \
  --config configs/stage5/stage5_eval.yaml \
  --checkpoint ${BEST_CKPT} \
  --split test \
  --run-name ${RUN_NAME}_test_eval \
  --device cuda
```

Blind eval:

```bash
python scripts/eval_stage5_blind_linepair.py \
  --config configs/stage5/stage5_blind_eval.yaml \
  --checkpoint ${BEST_CKPT} \
  --run-name ${RUN_NAME}_blind_eval \
  --device cuda
```

Expected output directories:
- training root: `outputs/stage5/main_run/${RUN_NAME}`
- val eval root: `outputs/stage5/eval/${RUN_NAME}_val_eval`
- test eval root: `outputs/stage5/eval/${RUN_NAME}_test_eval`
- blind eval root: `outputs/stage5/blind_eval/${RUN_NAME}_blind_eval`

Operational boundary:
- this repo state prepares the launch path only
- full long-run results remain pending actual Linux-server execution
- after the server run, record outcomes in `docs/execution/stage5_main_run_report.md`

### 7) Stage 6 controlled scale-fix run: L=5 with `gamma = 0.015`

This section defines a single-variable Stage 6 intervention on top of the
frozen Stage 5 paper path:

- keep `L=5` fixed
- keep dataset / optics / encoder / optimizer / step budget fixed
- change only the effective `L=5` gamma so `gamma_by_depth[5] = 0.015`

Dedicated configs:
- `configs/stage5/stage6_l5_gamma015_main.yaml`

Recommended shell variables on the Linux server:

```bash
GAMMA_L5=0.015
TOTAL_STEPS=10000
GAMMA_TAG=${GAMMA_L5/./p}
RUN_NAME=stage6_l5_scalefix_s${TOTAL_STEPS}_g5${GAMMA_TAG}
TRAIN_ROOT=outputs/stage6/scale_fix/${RUN_NAME}
BEST_CKPT=${TRAIN_ROOT}/checkpoints/checkpoint_best.pt
LATEST_CKPT=${TRAIN_ROOT}/checkpoints/checkpoint_latest.pt
```

Controlled-comparison note:
- this run is intended to be compared against the frozen Stage 5 main run
  `stage5_l5_phase_main`
- it is a Stage 6 debug intervention, not a replacement for the Stage 5 anchor

Training command:

```bash
python scripts/train_stage5_paper.py \
  --config configs/stage5/stage6_l5_gamma015_main.yaml \
  --steps ${TOTAL_STEPS} \
  --gamma-l5 ${GAMMA_L5} \
  --run-name ${RUN_NAME} \
  --device cuda \
  --download
```

If EMNIST is already present on the server, rerun the same command without
`--download`.

Parameter note:
- `steps` is controlled by `runtime.steps` in config and can be overridden by `--steps`
- `gamma_l5` is controlled by `experiment_params.gamma_l5` in config and can be overridden by `--gamma-l5`
- if you omit `--run-name`, the trainer will auto-build a name from the effective `steps` and `gamma_l5`
- `config_snapshot.json` records both effective values and whether they came from config or CLI

Resume command:

```bash
python scripts/train_stage5_paper.py \
  --config configs/stage5/stage6_l5_gamma015_main.yaml \
  --steps ${TOTAL_STEPS} \
  --gamma-l5 ${GAMMA_L5} \
  --device cuda \
  --resume ${LATEST_CKPT}
```

Regular eval on val:

```bash
python scripts/eval_stage5_paper.py \
  --config configs/stage5/stage5_eval.yaml \
  --checkpoint ${BEST_CKPT} \
  --split val \
  --run-name ${RUN_NAME}_val_eval \
  --device cuda
```

Regular eval on test:

```bash
python scripts/eval_stage5_paper.py \
  --config configs/stage5/stage5_eval.yaml \
  --checkpoint ${BEST_CKPT} \
  --split test \
  --run-name ${RUN_NAME}_test_eval \
  --device cuda
```

Blind eval:

```bash
python scripts/eval_stage5_blind_linepair.py \
  --config configs/stage5/stage5_blind_eval.yaml \
  --checkpoint ${BEST_CKPT} \
  --run-name ${RUN_NAME}_blind_eval \
  --device cuda
```

Expected output directories:
- training root: `outputs/stage6/scale_fix/${RUN_NAME}`
- val eval root: `outputs/stage5/eval/${RUN_NAME}_val_eval`
- test eval root: `outputs/stage5/eval/${RUN_NAME}_test_eval`
- blind eval root: `outputs/stage5/blind_eval/${RUN_NAME}_blind_eval`
