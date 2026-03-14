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
