# sdn-for-super-resolution-reproduction

This repository is a reproduction for the paper "Super-resolution image display using diffractive decoders".

This project uses a universal file structure.

The developer uses Codex to assist development.

## environment config

## orders for running

More orders are listed in docs/run_order.md. Here are some frequently used commands.

- Validate Stage 3 Issue 3 optical readout/crop:
  `python scripts/check_optical_readout.py`
- Validate Stage 3 Issue 4 optical forward sanity for L=1/3/5:
  `python scripts/check_optical_forward_depths.py`
- Run Stage 3 Issue 5 decoder-only single-sample fitting:
  `python scripts/train_decoder_only_single_sample.py --depth 3 --steps 150`
- Run Stage 3 Issue 6 decoder-only small-subset fitting:
  `python scripts/train_decoder_only_small_subset.py --depth 3 --subset-size 4 --steps 80`
- Run Stage 3 Issue 6 fixed-protocol depth sweep for L=1/3/5:
  `python scripts/train_decoder_only_small_subset_sweep.py --depths 1 3 5 --subset-size 4 --steps 80`
- Validate Stage 5 paper-aligned optics configs for L=1/3/5:
  `python scripts/check_stage5_optics_configs.py --config configs/stage5/stage5_optics_paper_aligned.yaml`
- Validate Stage 5 paper-aligned encoder + decoder integration:
  `python scripts/check_stage5_encoder_integration.py --config configs/stage5/stage5_paper_encoder.yaml`
- Validate Stage 5 paper-aligned SR loss:
  `python scripts/check_stage5_sr_loss.py --config configs/stage5/stage5_sr_loss.yaml`
- Run the Stage 5 paper-aligned short trainer sanity:
  `python scripts/train_stage5_paper.py --config configs/stage5/stage5_trainer_short.yaml`
- Preview the Stage 5 Issue 5.2 EMNIST 96x96 display dataset:
  `python scripts/preview_stage5_emnist_display.py --config configs/stage5/stage5_emnist_display.yaml`

