# Stage 3 Unresolved Parameters

## 1. 文档目的

本文件用于记录 Stage 3 planning freeze 之后的参数与设计决策状态：

- 已经冻结、可直接执行的 Stage-3 决策
- 仍未完全冻结的 paper / implementation 参数与设计点

本文件不会把尚未确认的事项伪装成既定事实；已经形成稳定结论的部分会明确标注为 finalized，尚未形成稳定结论的部分继续保留为 unresolved。

当前主要参考输入包括：

- `docs/integration/sdn_inventory.md`
- `docs/integration/sdn_optics_contract.md`
- `docs/integration/sdn_gap_analysis.md`
- `docs/plan/stage3_sdn_porting_plan.md`
- `docs/paper/paper_notes.md`

---

## 2. 分级规则

- `Finalized Stage-3 decisions`：已完成人工确认，后续实现必须直接遵循。
- `P1 — can finalize during implementation`：方向已基本明确，但具体工程落地方式可以在实现对应模块时定稿。
- `P2 — can defer until after first verification`：不影响 Stage 3 第一轮 forward sanity / decoder-only verification 启动，可在第一轮结果出来后再决定。

---

## 3. Finalized Stage-3 Decisions

以下 5 项 Stage-3 immediate decisions 已完成人工确认，当前状态统一视为 `RESOLVED`，后续实现不再将其作为 unresolved P0 项处理。

| 决策项 | 状态 | Finalized rule |
| --- | --- | --- |
| `L=1/3/5` 的 layer semantics | `RESOLVED` | `L` 定义为 trainable diffractive phase masks 数量，且 `L ∈ {1, 3, 5}`；最终传播到 sensor plane 不计入 layer count；传播距离由显式 distance schedule 控制，禁止使用 `layer == 2` 之类的隐式特判。 |
| distance representation | `RESOLVED` | Stage 3 固定采用带显式语义的 ordered distance schedule，并显式区分 `input_to_first`、`inter_layer`、`last_to_sensor`。 |
| ROI crop contract | `RESOLVED` | Stage 3 固定采用显式 center-crop ROI extraction；主链路为 `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi -> loss/metrics`；crop 逻辑不得隐藏在脚本中，且应由类似 `output_crop_hw` 的配置参数控制。 |
| full-grid vs ROI supervision contract | `RESOLVED` | Stage 3 固定采用 full-grid forward + ROI supervision：传播输出保留 `I_out_full`，loss 与主要 metrics 只在 `I_out_roi` 上计算，`I_out_full` 用于可视化、诊断和物理检查。 |
| normalized MAE default | `RESOLVED` | Stage 3 engineering default 为按单样本 ROI 像素计算 `sigma = sum(target) / sum(pred)`，分母加入 `epsilon` 保护，batch 结果按样本求平均；后续阶段可再做 paper-exact 对齐。 |

---

## 4. P1 — can finalize during implementation

| 参数项 | 当前状态 | 为什么 unresolved | 建议拍板时点 | 推荐默认决策 |
| --- | --- | --- | --- | --- |
| phase range mapping | 已确认主线是 phase-only，但还未冻结内部 raw parameter 如何映射到物理相位范围 | 论文没有把训练时的具体参数化约束写得足够细；upstream 也只提供了 wrapper 级约束思路 | 在 `phase_to_field` 或 phase parameter adapter 实现时拍板 | 保留显式 phase mapping 配置；默认把可训练 raw parameter 映射到物理 phase range，再生成 `U0 = exp(j * phi)` |
| intermediate field logging | 当前共识认为 forward sanity 需要可视化中间结果，但还未规定要保存哪些中间场、以什么粒度保存 | 过少的日志不利于排错，过多的日志会膨胀输出并模糊主链路 | 在 forward sanity script 实现时拍板 | 默认保存 `U_out_full`、`I_out_full`、`I_out_roi`，并将逐层中间场日志放在显式开关后面 |
| phase-mask initialization | `paper_notes.md` 已记录数值实验默认零初始化，但当前 Stage 3 还未规定是否只保留零初始化，还是同时暴露小随机初始化 | 这会影响 paper alignment 与实验弹性，但不阻塞 contract 本身 | 在 optics config 模板实现时拍板 | Stage 3 主线配置默认零初始化，同时允许把小随机初始化作为显式 ablation，而不是隐式替代主线 |

---

## 5. P2 — can defer until after first verification

| 参数项 | 当前状态 | 为什么 unresolved | 建议拍板时点 | 推荐默认决策 |
| --- | --- | --- | --- | --- |
| efficiency term on/off | 当前文档链已经确认它存在且并非所有实验都开启，但 Stage 3 第一轮还不需要先把它拍死 | 若在 normalized MAE 尚未稳定前就混入 efficiency term，会增加判断噪声 | 在第一次 forward sanity 和 decoder-only fitting 结果稳定后拍板 | 第一轮 verification 默认关闭 efficiency term；待主链路稳定后，再作为可开关 ablation 引入 |
| future quantization / misalignment extension hooks | 已知属于后续 paper alignment / robustness 方向，但当前没有必要定义过细接口 | 这些扩展不影响 Stage 3 主线是否成立，过早设计容易引入假需求 | 在第一次 verification 通过后，进入 Stage 5/6 规划时再拍板 | 当前只要求接口命名保持干净，不提前为 quantization 或 misalignment 写复杂扩展层 |

---

## 6. 使用说明

后续在拆 Stage 3 issue 或开始具体实现时，建议按以下顺序使用本文件：

1. 先按 Section 3 执行已经 finalized 的 Stage-3 决策。
2. 再将 `P1` 项绑定到对应实现 issue 中一并收敛。
3. 将 `P2` 项显式后置，不要在第一轮 verification 中偷偷引入。

本文件应随着 Stage 3 planning freeze 的推进而更新；已经 finalized 的规则应保持稳定，仍未确定的项目继续保留诚实标注。
