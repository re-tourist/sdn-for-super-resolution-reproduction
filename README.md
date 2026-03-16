# sdn-for-super-resolution-reproduction

​	This repository is a reproduction for paper "Super-resolution image display using diffractive decoders".

​	This project use a universal used file structure.

​	The developer use codex to assist in development.

## environment config

## orders for running

More order see docs/run_order.md. Here are some frequently used orders.

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

