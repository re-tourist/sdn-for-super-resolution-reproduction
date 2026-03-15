Task: implement Stage 3 Issue 6 — decoder-only small-subset optical capacity validation.

Context
-------
Stage 3 Issue 5 is already complete.

We now have a working decoder-only single-sample fitting entry that:
- does not use an encoder
- optimizes a learnable input phase pattern
- runs the current optical decoder forward path
- computes ROI-only normalized MAE
- has already shown clear loss decrease on at least one single target

Do NOT revisit Issue 5.
Do NOT redesign the decoder.
Do NOT introduce an encoder.
Do NOT start Stage 4 joint training.

Goal
----
Add a minimal, reproducible small-subset validation path to test whether
the current optical decoder stack shows consistent basic expressive capacity
across multiple targets, not just one single sample.

This is still a diagnostic Stage 3 issue, not a final experiment protocol.

Required behavior
-----------------
Implement a lightweight runnable entry such as:

- `scripts/train_decoder_only_small_subset.py`

or an equivalent minimal script.

The script should:

1. select a small deterministic subset of targets
   (for example 4, 8, or another small fixed number)
2. for each sample:
   - create a learnable input phase parameter
   - run decoder-only fitting
   - optimize for a limited number of steps
   - record initial / best / final loss
3. summarize whether fitting succeeds consistently across the subset
4. save enough artifacts or summaries to support diagnosis

Preferred scope
---------------
Keep this small and controlled.

Examples of acceptable outputs:
- a summary JSON/CSV over samples
- a compact per-sample artifact folder
- one or two aggregate visualizations if easy

What to verify
--------------
The script/output should let us check at minimum:

1. loss decreases for multiple targets, not just one
2. fitting is not restricted to a single lucky sample
3. ROI outputs change meaningfully during optimization
4. the current decoder-only path is reusable across samples
5. the issue remains fully encoder-free

Strong boundary
---------------
Do NOT:
- add encoder modules
- add Stage 4 minimal closed-loop training
- add full-dataset training
- redesign propagation core
- redesign readout/crop contract
- introduce quantization/misalignment/hardware robustness
- turn this into a full benchmark framework
- introduce paper-final hyperparameter sweeps

Depth handling
--------------
You may keep a fixed depth setting (for example L=3) for this issue
unless a tiny amount of depth configurability is already trivial.
Do NOT turn this issue into a full L=1/3/5 benchmark matrix.
That belongs later.

Loss / objective
----------------
Use the current Stage 3 engineering default normalized MAE on ROI.
No need to add PSNR/SSIM as the main optimization objective.

Implementation style
--------------------
This issue should extend the current single-sample fitting path in a clean,
minimal way, without overengineering.

If helpful, factor out a small reusable helper from the single-sample script,
but only if it keeps the code simpler.

Report back
-----------
After editing, report:

1. which files were modified
2. what runnable entry point was added
3. subset size and depth setting used
4. whether multiple samples showed decreasing loss
5. whether Issue 6 can be considered minimally complete