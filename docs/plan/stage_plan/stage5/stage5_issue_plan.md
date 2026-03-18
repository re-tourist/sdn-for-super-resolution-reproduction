**GPT-5.4**

下面给出一份 **Stage 5 详细工程推进计划**，用于直接拆成 GitHub issues 执行。内容基于当前仓库与既有文档：`PROJECT_CONTEXT.md`、`stage4_protocol_freeze.md`、`paper_notes.md`、`stage5_plan.md` 与 execution docs。

Stage 5 的核心不是“系统化消融”，而是 **把论文主设定落地成可运行、可复盘的工程协议**。系统化 sweep、量化、鲁棒性和大规模对比应后置到 Stage 6。

范围冲突提醒：

- 当 `paper_notes.md` 记录的是更宽的复现总范围，而 `stage5_plan.md` / 本文件记录的是当前阶段排期时，以 Stage 5 文档为准。
- 特别是 quantization、misalignment、systematic ablations 不应提前塞进 Stage 5 implementation prompt。

前置条件提醒：

- Stage 4 单样本 overfit 与小子集闭环已经通过
- Stage 4 regression baseline 只用于继承，不应在 Stage 5 issue 中重开

---

# 一、Stage 5 的工程目标重述

一句话版本：

> **把论文的完整设定（96×96 数据协议、200/400 光学网格、L=1/3/5、phase-only encoder、loss/效率项、训练超参、评估口径）落地为可复现 pipeline，并形成至少一个可真实启动的 paper-aligned 主线运行路径。**

---

# 二、Stage 5 拆解为 11 个 GitHub Issues

建议命名方式：

- Milestone: `Stage 5 — Paper-Aligned Settings`
- Issue 编号：`5.1 ~ 5.11`

---

# Issue 5.1 — Freeze Stage-5 paper-aligned protocol and unresolved-parameters ledger

## 这个 issue 要解决什么

先把 Stage 5 的协议冻住，不然所有实现都会漂。

要明确：

- 数据构造
- 光学 grid 与 distance mapping
- encoder 输出尺寸与 phase mapping
- loss 公式与 gamma 默认值
- 训练 / 评估协议
- 每个未决项由哪个后续 issue 拍板

## 产物

- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`

## 风险点

- 把 Stage 5 写成 Stage 6 的消融计划
- 让后续 issue 在错误位置擅自拍板未决项

## GitHub issue 正文建议

Title:
`Freeze Stage 5 paper-aligned protocol and unresolved-parameters ledger`

Body:
```md
## Background
Stage 5 should align the repo to the paper's full settings. Without a frozen protocol, implementations will drift and become untraceable.

## Suggested Branch
`docs/update`

## Goal
Write and freeze a Stage-5 protocol document that defines:
- paper-aligned data protocol
- optics grid/padding/distances
- encoder output contract
- loss and efficiency term
- training hyperparameters
- evaluation protocol
- explicit unresolved items
- ownership mapping from unresolved items to later issues

## Scope Priority
If `paper_notes.md` and Stage 5 planning docs disagree on stage scope, Stage 5 planning docs win.

## Tasks
- Review PROJECT_CONTEXT, Stage-4 protocol freeze, paper_notes, and stage5_plan
- Define the Stage 5 dataflow and tensor contract
- Freeze optics grid and distance schedule mapping rules
- Freeze loss definition and efficiency-term defaults as configurable policy
- Record all unresolved items explicitly
- Record which later issue is allowed to finalize each unresolved item

## Deliverable
- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`

## Acceptance
- The document makes Stage 5 scope explicit and prevents confusion with Stage 6
- Each unresolved item has an owner issue
- Future implementation issues can tell what they may or may not finalize
```

---

# Issue 5.2 — Build paper-aligned EMNIST 96×96 display dataset + augmentation

## 这个 issue 要解决什么

实现论文的 display dataset 构造：28×28 letters → 32×32 → 96×96 拼图，并对齐 train/val/test 划分与样本分布。

## 设计要点

- train/val 含 1~4 letters
- test 含 6~9 letters
- tile 规则、随机种子、空位策略必须可追溯
- augmentation 只做论文主线提到的 rotation / flip / contrast

## 产物

- 新数据模块（建议 `src/data/stage5_emnist_display.py`）
- 样本预览
- dataset config

## GitHub issue 正文建议

Title:
`Build paper-aligned 96×96 EMNIST display dataset with augmentation`

Body:
```md
## Background
Paper-aligned training requires the 96×96 EMNIST display dataset with specific train/val/test composition rules.

## Suggested Branch
`feat/data`

## Goal
Implement a dataset pipeline that reproduces the paper's 96×96 display protocol, including augmentation.

## Tasks
- Construct 96×96 samples from EMNIST letters (28→32, then tile)
- Enforce train/val/test size and letter-count distribution
- Add rotation/flip/contrast augmentation options
- Save a small preview grid of generated samples
- Keep tile rules and randomization explicitly configurable

## Non-Goals
- Do not introduce natural-image datasets
- Do not redesign Stage 4 data path
- Do not add Stage 6 quantization or robustness logic

## Deliverable
- Dataset module under `src/data/`
- Preview artifacts for sanity checks
- Config entries for dataset construction

## Acceptance
- A batch loads successfully with correct shapes
- Train/val/test splits match the paper's distribution
- Previews are visually interpretable and logged
```

---

# Issue 5.3 — Add paper-aligned optics configs for L=1/3/5 (200/400 grid + distances)

## 这个 issue 要解决什么

把论文的 optical geometry 变成可复用 config，并确保 L=1/3/5 都可 forward。

## 关键要点

- layer grid 200×200
- propagation grid 400×400（zero padding）
- input/output FOV 96×96
- distance schedule 显式映射到 `input_to_first / inter_layer / last_to_sensor`

## 风险点

- distance mapping 被悄悄拍板到代码里
- 顺手改动 optical core 设计

## GitHub issue 正文建议

Title:
`Add paper-aligned optics configs for L=1/3/5 (200/400 grid + distances)`

Body:
```md
## Background
Stage 5 requires paper-aligned optics geometry: 200×200 layer grid, 400×400 propagation grid, and explicit distance schedules.

## Suggested Branch
`feat/optics`

## Frozen Inheritance
- Keep the frozen optical contract:
  `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- Reuse the current optics hooks and module boundaries
- Treat Stage 4 artifacts as regression baseline, not as work to redo

## Goal
Create reusable optics configs for L=1/3/5 that match the paper settings and pass forward sanity checks.

## Allowed Decisions
- Express the paper distances as explicit config values
- Document the chosen mapping from paper notation to `input_to_first / inter_layer / last_to_sensor`

## Must Not Finalize Here
- Do not redefine the optical contract
- Do not add post-optics electrical heads
- Do not silently encode distance assumptions inside implementation logic

## Tasks
- Define paper-aligned grid/padding settings in config files
- Map paper distances to input_to_first / inter_layer / last_to_sensor
- Add or reuse a sanity script to validate shapes and forward stability
- Record the mapping rationale in docs or config comments

## Deliverable
- Optics config files under `configs/optics/`
- Forward sanity evidence for L=1/3/5

## Acceptance
- L=1/3/5 forward works with correct shapes
- Distance mapping is explicit and documented
- No optical-core redesign is introduced
```

---

# Issue 5.4 — Implement paper-aligned phase-only encoder (v1)

## 这个 issue 要解决什么

实现论文主线 encoder，把 96×96 HR 输入映射为低分辨率 phase-only `phi_lr`。

## 关键要点

- 输出 shape 对齐 paper LR pattern
- 相位范围映射必须显式
- 不能借机改 optical core

## GitHub issue 正文建议

Title:
`Implement paper-aligned phase-only encoder for Stage 5`

Body:
```md
## Background
Stage 5 requires a paper-aligned encoder that outputs a low-resolution phase-only pattern compatible with the frozen optical contract.

## Suggested Branch
`feat/model`

## Frozen Inheritance
- Keep the optical contract unchanged
- Reuse `forward_from_phase(...)` / current phase-provider-friendly integration path
- Stage 4 baseline remains the regression reference

## Goal
Add a paper-aligned encoder module that maps 96×96 HR input to a phase-only `phi_lr` tensor.

## Allowed Decisions
- Implement a configurable phase mapping
- Make output size configurable if the final paper-aligned size is still unresolved

## Must Not Finalize Here
- Do not claim one phase range is paper-uniquely settled if docs still mark it unresolved
- Do not modify optics-core files to accommodate encoder convenience

## Tasks
- Implement `paper_encoder_v1.py` under `src/models/encoder/`
- Ensure output spatial size matches the Stage 5 protocol choice
- Apply explicit phase-range mapping
- Keep the optical core untouched

## Deliverable
- Paper-aligned encoder module
- Config options for encoder size and phase mapping

## Acceptance
- Output shape and range are correct
- Forward integrates with the current diffractive decoder
- Unresolved phase-range choices stay configurable and logged
```

---

# Issue 5.5 — Implement paper-aligned loss with efficiency penalty

## 这个 issue 要解决什么

把论文的 normalized MAE + efficiency term 落地为可配置损失。

## 关键要点

- σ 归一化项按论文公式实现
- efficiency term 可开关
- γ 对 L=1/3/5 可配置

## GitHub issue 正文建议

Title:
`Add paper-aligned SR loss with efficiency penalty`

Body:
```md
## Background
The paper uses normalized MAE plus an efficiency term. Stage 5 must align this loss exactly and keep it configurable.

## Suggested Branch
`feat/train`

## Frozen Inheritance
- Keep the frozen optical readout contract
- Reuse the current Stage 4 understanding of normalized ROI supervision as historical context only

## Goal
Implement the paper-aligned loss function with a switchable efficiency term and configurable gamma values.

## Allowed Decisions
- Implement configurable gamma defaults per depth
- Record unresolved gamma policy in config / summary

## Must Not Finalize Here
- Do not hard-code an unreviewed L=5 gamma as if it were settled
- Do not bundle Stage 6 ablation logic into the loss implementation

## Tasks
- Implement normalized MAE with sigma normalization
- Add the efficiency penalty term
- Make gamma and on/off flags configurable per depth
- Add a small sanity check for numerical stability

## Deliverable
- Loss implementation under `src/losses/`
- Config options for gamma and efficiency toggle

## Acceptance
- Loss runs stably on a fake batch
- Gamma values and toggles are configurable and logged
- Unresolved gamma policy remains explicit
```

---

# Issue 5.6 — Build Stage-5 paper-aligned trainer + configs

## 这个 issue 要解决什么

实现 Stage 5 主线训练脚本，能跑 paper-aligned 设置并保存完整工件。

## 关键要点

- 支持 encoder / decoder 参数分组（不同 LR）
- 支持 resume
- 只为 Stage 5 paper path 服务，不做通用训练框架

## GitHub issue 正文建议

Title:
`Implement Stage 5 paper-aligned trainer and configs`

Body:
```md
## Background
Stage 5 needs a dedicated trainer that runs the paper-aligned pipeline with correct configs and artifacts.

## Suggested Branch
`feat/train`

## Frozen Inheritance
- Keep the optical contract unchanged
- Keep Stage 4 as regression baseline
- Reuse trusted training skeleton ideas where useful, without generalizing the whole repo

## Goal
Create a paper-aligned training script with clear configs, artifact saving, and resume support.

## Non-Goals
- Do not turn this into a general training framework
- Do not fill empty `scripts/train.py` as a global entrypoint
- Do not redesign Stage 4 trainer for reuse beyond what Stage 5 strictly needs

## Tasks
- Add `scripts/train_stage5_paper.py`
- Support separate LR for encoder and decoder
- Save config snapshots, summary, metrics, checkpoints, and previews
- Keep the script Stage-5-specific and paper-focused

## Deliverable
- Stage 5 trainer script
- Config templates under `configs/`

## Acceptance
- A short run completes successfully
- All expected artifacts are generated and logged
- The script remains Stage-5-specific rather than framework-generic
```

---

# Issue 5.7 — Add regular evaluation runner with bicubic baseline

## 这个 issue 要解决什么

实现 paper-aligned 常规评估：PSNR/SSIM 与 bicubic baseline。

## 关键要点

- 先把常规 eval 打通
- 不与 blind line-pair test 混在一个 issue 里

## GitHub issue 正文建议

Title:
`Add Stage 5 evaluation runner with bicubic baseline`

Body:
```md
## Background
Stage 5 needs a regular evaluation path for PSNR/SSIM and the paper's bicubic baseline before adding blind line-pair tests.

## Suggested Branch
`feat/eval`

## Goal
Provide a unified evaluation runner for regular val/test evaluation and bicubic baseline comparison.

## Tasks
- Implement `scripts/eval_stage5_paper.py`
- Add bicubic baseline generation with anti-aliasing
- Log PSNR/SSIM outputs in a reproducible format
- Record crop / normalization choices explicitly

## Non-Goals
- Do not add blind line-pair generation here
- Do not build a broad evaluation subsystem

## Deliverable
- Evaluation script
- Bicubic baseline utility

## Acceptance
- PSNR/SSIM can be computed on val/test
- Bicubic baseline is reproducible
- Metric outputs and evaluation assumptions are logged
```

---

# Issue 5.8 — Add blind line-pair generator and eval hook

## 这个 issue 要解决什么

把 blind line-pair / resolution target 测试独立出来，避免和常规 eval 混成一个大任务。

## GitHub issue 正文建议

Title:
`Add Stage 5 blind line-pair generator and evaluation hook`

Body:
```md
## Background
Blind line-pair testing is important for paper alignment, but it should be implemented as a separate scoped task instead of being bundled into the regular evaluation runner.

## Suggested Branch
`feat/eval`

## Goal
Add a blind line-pair / resolution-target generation path and hook it into the Stage 5 evaluation workflow.

## Tasks
- Add line-pair / resolution-target generator utilities
- Define how blind-test outputs are stored and compared
- Hook blind-test execution into the evaluation path without rewriting regular eval
- Document assumptions about target generation and reporting

## Non-Goals
- Do not redesign the regular evaluation runner
- Do not introduce quantization or robustness studies here

## Deliverable
- Blind line-pair utility / hook
- Logged blind-test artifact path and reporting convention

## Acceptance
- Blind line-pair tests can be generated and executed
- Artifacts are saved in a reproducible format
- The implementation stays separate from regular eval concerns
```

---

# Issue 5.9 — Run paper-aligned smoke training and write report

## 这个 issue 要解决什么

用 paper-aligned 配置做一次短训 smoke，验证 pipeline 稳定。

## GitHub issue 正文建议

Title:
`Run paper-aligned smoke training and write Stage 5 smoke report`

Body:
```md
## Background
Before launching long runs, we need a paper-aligned smoke run to validate the pipeline end-to-end.

## Suggested Branch
`feat/train`

## Goal
Run a short paper-aligned training and record stability and artifacts.

## Tasks
- Run a short smoke training with paper configs
- Save checkpoints and preview outputs
- Record loss trends and stability findings
- Write a concise smoke report

## Non-Goals
- Do not treat smoke as the final paper result
- Do not expand this into main-run tuning

## Deliverable
- Smoke run outputs under `outputs/stage5/`
- `docs/execution/stage5_smoke_report.md`

## Acceptance
- Training completes without NaN/Inf
- Loss decreases at least modestly
- Artifacts and logs are complete
```

---

# Issue 5.10 — Run paper-aligned main training (phase-only, L=5) + report

## 这个 issue 要解决什么

执行 paper-aligned 主线训练（建议 L=5 phase-only），并记录结果。

## 关键要点

- 这是 main-run issue，不是 trainer / eval 功能开发 issue
- 必须把“代码已就绪”与“长训资源是否足够”分开记录

## GitHub issue 正文建议

Title:
`Run paper-aligned main training (phase-only, L=5) and write report`

Body:
```md
## Background
Stage 5 should produce a paper-aligned main run once the pipeline, trainer, and evaluation path are already in place.

## Suggested Branch
`feat/train`

## Goal
Run the paper-aligned main training for phase-only L=5 and document the results.

## Tasks
- Launch the paper-aligned L=5 config
- Run evaluation on val/test using the existing Stage 5 eval path
- Compare against bicubic baseline
- Write a main-run report with assumptions, results, and remaining gaps

## Resource Boundary
- Distinguish clearly between:
  - code/config path is launch-ready
  - long-run budget was or was not fully available
- If compute limits block a full target-budget run, record that as a resource constraint, not as a hidden implementation detail

## Non-Goals
- Do not reopen trainer architecture here
- Do not add Stage 6 sweeps or ablations here

## Deliverable
- Main run outputs under `outputs/stage5/`
- `docs/execution/stage5_main_run_report.md`

## Acceptance
- A real paper-aligned main run is launched and logged
- Evaluation metrics are recorded in a comparable format
- The report separates implementation readiness from resource-limited execution state
```

---

# Issue 5.11 — Consolidate Stage-5 docs and make Stage-6 go/no-go decision

## 这个 issue 要解决什么

把 Stage 5 执行结果收口，更新文档，并明确是否进入 Stage 6。

## GitHub issue 正文建议

Title:
`Consolidate Stage 5 results and make Stage-6 go/no-go decision`

Body:
```md
## Background
Stage 5 is only useful if its results are consolidated and used to decide Stage 6 entry.

## Suggested Branch
`docs/update`

## Goal
Summarize Stage 5 execution, update docs, and make an explicit Stage-6 decision.

## Tasks
- Update experiment log and results summary
- Record all paper-aligned assumptions and unresolved-item outcomes
- Summarize smoke and main-run outcomes
- Make a clear go/no-go call for Stage 6

## Deliverable
- Updated docs with a clear Stage 6 decision

## Acceptance
- The repo states clearly whether Stage 5 passed
- The docs distinguish engineering completion from resource-limited run status
- Stage 6 entry conditions are explicit
```

---

# 三、推荐执行顺序（依赖优先）

1. Issue 5.1
2. Issue 5.2 + 5.3
3. Issue 5.4 + 5.5
4. Issue 5.6
5. Issue 5.7
6. Issue 5.8
7. Issue 5.9
8. Issue 5.10
9. Issue 5.11

---

# 四、Milestone 建议

Milestone：`Stage 5 — Paper-Aligned Settings`

Issues：

- 5.1 Freeze Stage 5 paper-aligned protocol and unresolved-parameters ledger
- 5.2 Build paper-aligned 96×96 EMNIST display dataset with augmentation
- 5.3 Add paper-aligned optics configs for L=1/3/5 (200/400 grid + distances)
- 5.4 Implement paper-aligned phase-only encoder for Stage 5
- 5.5 Add paper-aligned SR loss with efficiency penalty
- 5.6 Implement Stage 5 paper-aligned trainer and configs
- 5.7 Add Stage 5 evaluation runner with bicubic baseline
- 5.8 Add Stage 5 blind line-pair generator and evaluation hook
- 5.9 Run paper-aligned smoke training and write Stage 5 smoke report
- 5.10 Run paper-aligned main training (phase-only, L=5) and write report
- 5.11 Consolidate Stage 5 results and make Stage-6 go/no-go decision
