**GPT-5.4**

下面给出一份 **Stage 5 详细工程推进计划**，用于直接拆成 GitHub issues 执行。内容基于当前仓库与既有文档：`plan_overview.md`、`stage4_protocol_freeze.md`、`paper_notes.md` 和 execution logs。

Stage 5 的核心不是“系统化消融”，而是 **把论文主设定落地成可运行、可复盘的工程协议**。系统化 sweep、量化、鲁棒性和大规模对比应后置到 Stage 6。

前置条件提醒：

- Stage 4 单样本 overfit 与小子集闭环必须已经通过
- 若单样本闭环报告缺失，建议先补齐，再进入 Stage 5

---

# 一、Stage 5 的工程目标重述

一句话版本：

> **把论文的完整设定（96×96 数据协议、200/400 光学网格、L=1/3/5、phase-only encoder、loss/效率项、训练超参、评估口径）落地为可复现 pipeline，并产出首个 paper-aligned 训练结果。**

---

# 二、Stage 5 拆解为 10 个 GitHub Issues

建议命名方式：

- Milestone: `Stage 5 — Paper-Aligned Settings`
- Issue 编号：`5.1 ~ 5.10`

---

# Issue 5.1 — Freeze Stage-5 paper-aligned protocol and unresolved-parameters ledger

## 这个 issue 要解决什么

先把 Stage 5 的协议冻住，不然所有实现都会漂。

要明确：数据构造、光学 grid、distance schedule、encoder 输出尺寸、loss 公式、训练超参与评估协议，并把未决项列成清单。

## 输入

- `docs/plan/plan_overview.md`
- `docs/plan/stage_plan/stage4/stage4_protocol_freeze.md`
- `docs/paper/paper_notes.md`
- `docs/execution/results_summary.md`

## 产物

- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`

## 验收标准

- 文档明确 Stage 5 与 Stage 6 的边界
- 论文设定中的关键参数都写清楚
- 所有不确定项以“待确认”列出

## 风险点

最大风险是把 Stage 5 写成 Stage 6 的消融计划。

## GitHub issue 正文建议

Title:
`Freeze Stage 5 paper-aligned protocol and unresolved-parameters ledger`

Body:
```md
## Background
Stage 5 should align the repo to the paper's full settings. Without a frozen protocol, implementations will drift and become untraceable.

## Goal
Write and freeze a Stage-5 protocol document that defines:
- paper-aligned data protocol
- optics grid/padding/distances
- encoder output contract
- loss and efficiency term
- training hyperparameters
- evaluation protocol
- explicit unresolved items

## Tasks
- Review plan_overview, Stage-4 protocol freeze, and paper_notes
- Define the Stage 5 dataflow and tensor contract
- Freeze optics grid and distance schedule mapping
- Freeze loss definition and efficiency term defaults
- Record all unresolved items explicitly

## Deliverable
- `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`

## Acceptance
The document makes Stage 5 scope explicit and prevents confusion with Stage 6.
```

---

# Issue 5.2 — Build paper-aligned EMNIST 96×96 display dataset + augmentation

## 这个 issue 要解决什么

实现论文的 display dataset 构造：28×28 letters → 32×32 → 96×96 拼图，并对齐 train/val/test 划分与样本分布。

## 设计要点

- train/val 含 1~4 letters
- test 含 6~9 letters
- 记录 tile 规则与随机种子
- 加入 rotation / flip / contrast augmentation

## 产物

- 新数据模块（建议 `src/data/stage5_emnist_display.py`）
- 可视化样本预览
- config 入口（不把参数写死）

## 验收标准

- 可生成 96×96 图像 batch
- train/val/test 分布符合论文描述
- 保存预览图可解释

## 风险点

tile 规则若不透明，后续结果无法对齐论文。

## GitHub issue 正文建议

Title:
`Build paper-aligned 96×96 EMNIST display dataset with augmentation`

Body:
```md
## Background
Paper-aligned training requires the 96×96 EMNIST display dataset with specific train/val/test composition rules.

## Goal
Implement a dataset pipeline that reproduces the paper's 96×96 display protocol, including augmentation.

## Tasks
- Construct 96×96 samples from EMNIST letters (28→32, then tile)
- Enforce train/val/test size and letter-count distribution
- Add rotation/flip/contrast augmentation options
- Save a small preview grid of generated samples
- Keep all rules configurable

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

把论文的 optical geometry 变成可复用的 config，并确保 L=1/3/5 都可 forward。

## 关键要点

- layer grid 200×200
- propagation grid 400×400（zero padding）
- input/output FOV 96×96
- 距离 schedule 必须显式映射到 `input_to_first / inter_layer / last_to_sensor`

## 产物

- optics config（建议 `configs/optics/paper_l1.yaml`、`paper_l3.yaml`、`paper_l5.yaml`）
- smoke check 脚本或更新现有 sanity script

## 验收标准

- L=1/3/5 forward 成功
- 输出 shape 与 ROI 对齐
- distances mapping 清楚且可追溯

## 风险点

距离映射不清会导致“跑得通但不对齐”。

## GitHub issue 正文建议

Title:
`Add paper-aligned optics configs for L=1/3/5 (200/400 grid + distances)`

Body:
```md
## Background
Stage 5 requires paper-aligned optics geometry: 200×200 layer grid, 400×400 propagation grid, and explicit distance schedules.

## Goal
Create reusable optics configs for L=1/3/5 that match the paper settings and pass forward sanity checks.

## Tasks
- Define paper-aligned grid/padding settings in config files
- Map paper distances to input_to_first / inter_layer / last_to_sensor
- Add or reuse a sanity script to validate shapes and forward stability

## Deliverable
- Optics config files under `configs/optics/`
- Forward sanity evidence for L=1/3/5

## Acceptance
- L=1/3/5 forward works with correct shapes
- Distance mapping is explicit and documented
```

---

# Issue 5.4 — Implement paper-aligned phase-only encoder (v1)

## 这个 issue 要解决什么

实现论文主线 encoder，把 96×96 HR 输入映射为低分辨率 phase-only `phi_lr`。

## 设计要点

- 输出 shape 对齐 paper LR pattern（默认 32×32，但需在协议中确认）
- 显式相位范围映射（例如 `[-π, π]`）
- 不改 optical core

## 产物

- `src/models/encoder/paper_encoder_v1.py`
- 对应 config 入口

## 验收标准

- forward 输出 shape 正确
- phase 范围受控
- 可接入 `forward_from_phase(...)`

## 风险点

输出尺寸或相位范围不明确会导致全链路偏差。

## GitHub issue 正文建议

Title:
`Implement paper-aligned phase-only encoder for Stage 5`

Body:
```md
## Background
Stage 5 requires a paper-aligned encoder that outputs a low-resolution phase-only pattern compatible with the frozen optical contract.

## Goal
Add a paper-aligned encoder module that maps 96×96 HR input to a phase-only `phi_lr` tensor.

## Tasks
- Implement `paper_encoder_v1.py` under `src/models/encoder/`
- Ensure output spatial size matches the paper LR pattern size
- Apply explicit phase-range mapping
- Keep the optical core untouched

## Deliverable
- Paper-aligned encoder module
- Config options for encoder size and phase mapping

## Acceptance
- Output shape and range are correct
- Forward integrates with the current diffractive decoder
```

---

# Issue 5.5 — Implement paper-aligned loss with efficiency penalty

## 这个 issue 要解决什么

把论文的 normalized MAE + efficiency term 落地为可配置损失。

## 设计要点

- σ 归一化项按论文公式实现
- efficiency term 可开关
- γ 对 L=1/3/5 可配置

## 产物

- loss 实现（建议 `src/losses/paper_sr_loss.py`）
- config 入口
- 最小 unit check（前向数值稳定）

## 验收标准

- loss 数值稳定（无 NaN/Inf）
- γ 与开关可配置

## 风险点

若 σ 归一化实现偏差，会导致训练曲线不可对齐。

## GitHub issue 正文建议

Title:
`Add paper-aligned SR loss with efficiency penalty`

Body:
```md
## Background
The paper uses normalized MAE plus an efficiency term. Stage 5 must align this loss exactly and keep it configurable.

## Goal
Implement the paper-aligned loss function with a switchable efficiency term and configurable gamma values.

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
```

---

# Issue 5.6 — Build Stage-5 paper-aligned trainer + configs

## 这个 issue 要解决什么

实现 Stage 5 主线训练脚本，能跑 paper-aligned 设置并保存完整工件。

## 设计要点

- 支持 encoder / decoder 参数分组（不同 LR）
- 支持长训与 resume
- 保存 config snapshot / summary / metrics / checkpoints / sample previews

## 产物

- `scripts/train_stage5_paper.py`
- 训练 config 模板（dataset / optics / loss / optimizer）

## 验收标准

- 能完成短 smoke 训练
- 产物齐全且可追溯

## 风险点

trainer 过度泛化会把 Stage 6 需求提前塞进来。

## GitHub issue 正文建议

Title:
`Implement Stage 5 paper-aligned trainer and configs`

Body:
```md
## Background
Stage 5 needs a dedicated trainer that runs the paper-aligned pipeline with correct configs and artifacts.

## Goal
Create a paper-aligned training script with clear configs, artifact saving, and resume support.

## Tasks
- Add `scripts/train_stage5_paper.py`
- Support separate LR for encoder and decoder
- Save config snapshots, summary, metrics, checkpoints, and previews
- Keep the script minimal and paper-focused

## Deliverable
- Stage 5 trainer script
- Config templates under `configs/`

## Acceptance
- A short run completes successfully
- All expected artifacts are generated and logged
```

---

# Issue 5.7 — Add evaluation runner + bicubic baseline + line-pair test

## 这个 issue 要解决什么

实现 paper-aligned 评估：PSNR/SSIM、bicubic baseline、blind line-pair test。

## 产物

- `scripts/eval_stage5_paper.py`
- bicubic baseline 生成逻辑
- line-pair / resolution target 生成与评估入口

## 验收标准

- 可以对 val/test 集输出 PSNR/SSIM
- baseline 结果可复盘
- line-pair 测试可运行

## 风险点

评估口径不一致会导致结果不可比较。

## GitHub issue 正文建议

Title:
`Add Stage 5 evaluation runner with bicubic baseline and line-pair test`

Body:
```md
## Background
Paper alignment requires consistent evaluation: PSNR/SSIM, bicubic baseline, and blind line-pair tests.

## Goal
Provide a unified evaluation runner that reproduces paper-aligned metrics and baselines.

## Tasks
- Implement `scripts/eval_stage5_paper.py`
- Add bicubic baseline generation (with anti-aliasing)
- Add line-pair / resolution target generator and eval hook
- Log metric outputs in a reproducible format

## Deliverable
- Evaluation script
- Baseline and line-pair utilities

## Acceptance
- PSNR/SSIM can be computed on val/test
- Bicubic baseline is reproducible
- Line-pair tests can be executed and logged
```

---

# Issue 5.8 — Run paper-aligned smoke training and write report

## 这个 issue 要解决什么

用 paper-aligned 配置做一次短训 smoke，验证 pipeline 稳定。

## 产物

- `outputs/stage5/paper_smoke/...`
- `docs/execution/stage5_smoke_report.md`

## 验收标准

- 无 NaN/Inf
- loss 有下降趋势
- 训练产物齐全

## 风险点

若 smoke 不通过，应先修 pipeline，不进入主线大训。

## GitHub issue 正文建议

Title:
`Run paper-aligned smoke training and write Stage 5 smoke report`

Body:
```md
## Background
Before launching long runs, we need a paper-aligned smoke run to validate the pipeline end-to-end.

## Goal
Run a short paper-aligned training and record stability and artifacts.

## Tasks
- Run a short smoke training with paper configs
- Save checkpoints and preview outputs
- Record loss trends and stability findings
- Write a concise smoke report

## Deliverable
- Smoke run outputs under `outputs/stage5/`
- `docs/execution/stage5_smoke_report.md`

## Acceptance
- Training completes without NaN/Inf
- Loss decreases at least modestly
- Artifacts and logs are complete
```

---

# Issue 5.9 — Run paper-aligned main training (phase-only, L=5) + report

## 这个 issue 要解决什么

执行 paper-aligned 主线训练（建议 L=5 phase-only），并记录结果。

## 产物

- `outputs/stage5/paper_main_l5/...`
- `docs/execution/stage5_main_run_report.md`

## 验收标准

- 训练完成并生成完整工件
- PSNR/SSIM 与 baseline 可对比
- 未决项在报告中被明确记录

## 风险点

训练成本高，需提前评估算力与时间预算。

## GitHub issue 正文建议

Title:
`Run paper-aligned main training (phase-only, L=5) and write report`

Body:
```md
## Background
Stage 5 should produce at least one paper-aligned main run to anchor the reproduction results.

## Goal
Run the paper-aligned main training for phase-only L=5 and document the results.

## Tasks
- Train with the paper-aligned L=5 config
- Run evaluation on val/test
- Compare against bicubic baseline
- Write a main-run report with assumptions and gaps

## Deliverable
- Main run outputs under `outputs/stage5/`
- `docs/execution/stage5_main_run_report.md`

## Acceptance
- Training completes with full artifacts
- Evaluation metrics are recorded and comparable
- Report clearly states assumptions and remaining gaps
```

---

# Issue 5.10 — Consolidate Stage-5 docs and make Stage-6 go/no-go decision

## 这个 issue 要解决什么

把 Stage 5 执行结果收口，更新文档，并明确是否进入 Stage 6。

## 产物

- 更新 `docs/execution/experiment_log.md`
- 更新 `docs/execution/results_summary.md`
- 更新 `docs/plan/reproduction_plan.md`

## 验收标准

- Stage 5 结果与假设被清楚记录
- Stage 6 进入条件被明确判定

## GitHub issue 正文建议

Title:
`Consolidate Stage 5 results and make Stage-6 go/no-go decision`

Body:
```md
## Background
Stage 5 is only useful if its results are consolidated and used to decide Stage 6 entry.

## Goal
Summarize Stage 5 execution, update docs, and make an explicit Stage-6 decision.

## Tasks
- Update experiment log and results summary
- Record all paper-aligned assumptions and unresolved items
- Summarize smoke and main-run outcomes
- Make a clear go/no-go call for Stage 6

## Deliverable
- Updated docs with a clear Stage 6 decision

## Acceptance
The repo states clearly whether Stage 5 passed and whether Stage 6 may start.
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
- 5.7 Add Stage 5 evaluation runner with bicubic baseline and line-pair test
- 5.8 Run paper-aligned smoke training and write Stage 5 smoke report
- 5.9 Run paper-aligned main training (phase-only, L=5) and write report
- 5.10 Consolidate Stage 5 results and make Stage-6 go/no-go decision
