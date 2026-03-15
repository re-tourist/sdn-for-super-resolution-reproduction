Task: implement Stage 3 Issue 4 — forward sanity checks for L=1/3/5.

Context
-------
Stage 3 Issue 3 is already complete.

The decoder contract is now:

phi_lr
 -> U0
 -> U_out_full
 -> I_out_full
 -> I_out_roi

Issue 3 readout/crop validation has already been closed out with a reusable script.
Do NOT revisit Issue 3.
Do NOT redesign decoder interfaces unless absolutely necessary for sanity coverage.

Current Stage 3 frozen decisions include:
- L means the number of trainable diffractive phase masks
- propagation distances use ordered schedule with explicit semantics
- Stage 3 uses explicit center-crop ROI
- supervision is ROI-based (but Issue 4 is not a training issue)
- normalized MAE belongs to later issue(s), not this one

Goal
----
Add a minimal, reproducible forward-sanity validation entry for L=1/3/5
to verify that the current paper-aligned optical decoder skeleton behaves
consistently across supported depths.

What Issue 4 should validate
----------------------------
For L in {1, 3, 5}, verify at minimum:

1. forward pass runs successfully
2. returned outputs include:
   - U_out_full
   - I_out_full
   - I_out_roi
3. U_out_full remains complex
4. I_out_full and I_out_roi remain real and nonnegative
5. ROI shape matches configured crop size
6. output interfaces are consistent across depths
7. outputs for different L are not trivially identical
8. current distance schedule / layer semantics do not rely on old implicit rules
   like hard-coded layer-index special cases

Preferred implementation style
------------------------------
Add a lightweight runnable validation asset, such as:

- `scripts/check_optical_forward_depths.py`

and only make minimal implementation/documentation changes needed to support it.

The script should:
- instantiate the current decoder in L=1, L=3, and L=5 modes
- run forward on a controlled phase input
- print or assert the key sanity results
- fail loudly on invalid shape/dtype/nonnegativity/depth behavior

Optional but acceptable
-----------------------
- save a very small number of debug figures or summaries
- add a short README note pointing to the new script

Not in scope
------------
Do NOT:
- add training loops
- add decoder-only fitting
- add loss integration
- add PSNR/SSIM
- add Stage 4 encoder integration
- redesign the propagation core beyond what is minimally required
- expand into quantization/misalignment/hardware robustness

Important boundary
------------------
Issue 4 is a forward-sanity issue, not a performance issue.
It only needs to prove that the current optical decoder skeleton is
stable, depth-aware, and contract-consistent for L=1/3/5.

Report back
-----------
After editing, report:

1. which files were modified
2. what sanity entry point was added
3. whether L=1/3/5 all ran successfully
4. whether all required sanity checks passed
5. whether Issue 4 can now be considered cleanly closed