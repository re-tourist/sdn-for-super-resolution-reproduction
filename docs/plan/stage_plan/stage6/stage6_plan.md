# 第六阶段具体规划（Stage 6 — 系统性排错、语义审计与总结）

## 1. 文档定位

本文档重新定义本仓库的 **Stage 6**。

Stage 5 已经完成 paper-aligned pipeline 的工程落地，但主线 `L=5 phase-only` 结果明显差于 bicubic baseline，且 `preview_final.png` 暴露出原始 `pred_roi` 强度塌缩问题。基于 `docs/execution/troubleshooting.md` 中的 `Issue T003`，Stage 6 的首要任务不再是大规模系统化实验，而是：

- 系统性排错
- loss / eval 语义审计
- 失败模式分解
- 窄范围、可解释的定向对照

只有在主失败机制被澄清后，才恢复更宽的 ablation / sweep。

本文件优先级低于：

1. `AGENTS.md`
2. `docs/ai/PROJECT_CONTEXT.md`
3. `docs/plan/stage_plan/stage5/stage5_protocol_freeze.md`
4. `docs/execution/stage5_main_run_report.md`
5. `docs/execution/troubleshooting.md`

如与旧的 Stage 6 “系统化实验”表述冲突，以当前文档为准。

---

## 2. 为什么要把 Stage 6 改成系统性排错

当前证据已经足够支持下面 3 个判断：

1. **scale ambiguity 是真实机制**。
   - Stage 5 主损失使用 `|y - sigma * y_hat|`。
   - 当前主线 `L=5` 又配置为 `gamma_by_depth[5] = 0.0`。
   - 因此 loss 对 raw 输出的全局强度缺乏约束，模型可以落到极低能量解。

2. **scale ambiguity 还不能被宣布为唯一根因**。
   - raw `pred_roi` 接近全黑，解释了 “为什么 preview 黑”。
   - 但还没有充分解释 “为什么 regular eval / blind eval 仍明显输给 bicubic”。

3. **raw 与 sigma-rescaled 指标几乎一致，是必须优先审计的异常信号**。
   - 这提示至少要先排查：
     - eval 指标是否对幅值不敏感
     - 指标路径是否存在隐式归一化
     - 即使解决强度塌缩，结构重建是否依旧失败

因此，ChatGPT 反馈中的以下建议是可取的，并被 Stage 6 采纳：

- 不把 “scale ambiguity” 过度表述成唯一根因
- 先做 `metric / eval semantics audit`
- 明确拆分两类失败：
  - 问题 A：raw intensity collapse
  - 问题 B：normalized reconstruction failure
- 保留 raw preview，同时增加 sigma-aware 诊断视图
- 用定向、小预算 ablation 替代大 sweep

以下建议只部分采纳：

- “显式全局 gain 参数” 是一个值得测试的干预方案，但在 Stage 6 中应作为受控对照之一，而不是默认正确修法。

---

## 3. Stage 6 需要回答的核心问题

Stage 6 不是继续堆实验，而是要把当前主结果失败拆解成可回答的问题：

1. 当前 regular eval / blind eval 到底在比较什么？
   - raw tensor？
   - 隐式归一化后的图？
   - 是否对全局幅值缩放近似不敏感？

2. 当前失败主要来自哪一类？
   - A：raw 强度塌缩
   - B：归一化后结构仍然差
   - A 与 B 同时存在

3. 失败是否主要与 `L=5` 相关？
   - 是 depth-specific failure，还是 paper-aligned path 的共性问题？

4. 如果 scale ambiguity 被部分修复，指标能否改善？
   - 若 raw 恢复但 normalized 结构仍差，则要继续排查 geometry / phase / optimization 问题。

5. 是否需要重新打开 Stage 5 中的某些 paper-sensitive 假设？
   - distance mapping
   - phase range
   - crop / FOV alignment
   - 其它 physics-sensitive config

---

## 4. Stage 6 非目标

以下内容不应在 Stage 6 初期直接展开：

- 大范围 `L=1/3/5` 完整 sweep
- quantization 全量 bit-width study
- misalignment robustness 矩阵
- phase-only / complex-valued / amplitude-only 的大对照
- 没有明确假设支撑的参数扫表
- 重写 Stage 5 trainer / eval framework
- 悄悄修改 Stage 5 protocol 并把它包装成“修 bug”

这些工作只有在核心失败模式解释清楚后，才允许恢复。

---

## 5. 失败分类与证据分层

### 5.1 两类主失败模式

**问题 A：raw intensity collapse**

表现：

- raw `pred_roi` 绝对强度极低
- `prediction_sum << target_sum`
- raw preview 在固定 `[0, 1]` 下近乎全黑

**问题 B：normalized reconstruction failure**

表现：

- 即使做 `sigma * pred_roi` 或其它显式重标定
- `PSNR / SSIM` 仍明显差于 bicubic
- 结构、高频或边缘恢复仍然失败

Stage 6 所有报告都必须显式区分 A 和 B，不能把它们混成一句 “模型效果不好”。

### 5.2 证据分层

Stage 6 采用下面的证据梯度：

1. **S0 — 语义审计**：不训练，只澄清 loss / eval / preview 的真实语义。
2. **S1 — 观测增强**：增加 raw / rescaled / normalized 视图与统计。
3. **S2 — 极小规模复现**：用 tiny-set / small-set 复现实效模式。
4. **S3 — 单变量干预**：一次只改一个核心变量，验证是否修复 A 或 B。
5. **S4 — 协议重开**：只有在 S0~S3 不能解释现象时，才允许重新审计 paper-sensitive config。

---

## 6. Stage 6 工作流

### 6.1 P0：语义审计

必须优先完成：

- audit `PSNR / SSIM` 的输入语义
- audit bicubic baseline 的 resize / antialias / normalization 路径
- audit blind eval 是否引入额外归一化或只保留形状信息
- 用最简单的缩放单元测试验证 metric 是否对全局强度敏感

P0 的目标不是提高指标，而是先让后续实验“可解释”。

### 6.2 P1：可观测性增强

在训练、eval、报告中同时记录：

- raw `pred_roi`
- `sigma * pred_roi`
- 必要时的 per-sample normalized `pred_roi`
- raw `prediction_sum`
- target `sum`
- `sigma` 的 min / max / mean
- raw / rescaled 的 `PSNR / SSIM`

P1 的目标是把 A 和 B 清楚分开。

### 6.3 P2：极小规模复现

最优先的不是大训练，而是高信息密度的小实验：

- `1~4` 样本 tiny-set overfit
- 固定小预算 small-subset rerun
- 同一协议下比较 `L=5` 与较浅 depth

P2 的目标是判断：

- 问题是 loss identifiability、优化、还是 physics config
- 问题是在 tiny-set 就存在，还是只在更大数据上才出现

### 6.4 P3：定向小型干预

只允许测试有明确假设支持的窄修改，例如：

- 给 `L=5` 加弱能量约束
- 测试非零 `gamma` 对 raw 输出的影响
- 显式引入全局 gain 参数

要求：

- 一次只改一个主变量
- 必须保留 Stage 5 baseline 作为 anchor
- 必须报告是否只修复 A，还是同时改善 B

### 6.5 P4：条件性协议复审

只有在下面情况同时成立时，才允许重开 paper-sensitive 假设：

- eval semantics 已经审计清楚
- scale-related 干预不能解释主要差距
- tiny-set / small-set 仍显示结构性失败

此时才进入对下列项的受控复审：

- distance mapping
- phase range
- crop / FOV alignment
- 其它 physics-sensitive defaults

---

## 7. Stage 6 具体任务族

### 7.1 语义与协议

- 冻结 Stage 6 debug protocol
- 审计 eval / blind eval / metric invariance
- 明确哪些 issue 可以改 loss，哪些只能加日志

### 7.2 观测与诊断

- 增加 sigma-aware preview
- 报告 raw / rescaled / normalized 三类视图
- 增加 `prediction_sum / target_sum / sigma` 统计

### 7.3 受控复现

- tiny-set overfit on `L=5`
- fixed-budget small-subset rerun on `L=5`
- 视需要比较 `L=3` 与 `L=5`

### 7.4 窄范围干预

- 弱能量约束
- `gamma` 策略对照
- 可选的显式全局 gain 参数

### 7.5 条件性 physics 复审

- 仅在前面工作不能解释结果时进行
- 不允许一上来大改 geometry

### 7.6 文档收口

- 更新 `troubleshooting.md`
- 更新 `results_summary.md`
- 更新 `experiment_log.md`
- 明确是否恢复更宽的 systematic experiments

---

## 8. 推荐 issue 顺序

推荐顺序如下：

1. `Issue 6.1` — Freeze Stage 6 debug protocol and failure taxonomy
2. `Issue 6.2` — Audit regular/blind eval semantics and metric scale sensitivity
3. `Issue 6.3` — Add sigma-aware diagnostics and preview/reporting upgrades
4. `Issue 6.4` — Reproduce the `L=5` failure on tiny-set and small-subset budgets
5. `Issue 6.5` — Test narrow scale-fixing interventions on the frozen `L=5` path
6. `Issue 6.6` — Compare `L=3` and `L=5` under the same debug protocol
7. `Issue 6.7` — Re-audit paper-sensitive geometry assumptions only if needed
8. `Issue 6.8` — Consolidate Stage 6 debugging findings and redefine post-Stage-6 policy

原因：

- `6.1 ~ 6.3` 决定后续实验是否可解释
- `6.4 ~ 6.5` 直接验证当前最强假设
- `6.6` 用来判断问题是否 depth-specific
- `6.7` 只在必要时才重开 physics 假设
- `6.8` 负责把调试结果沉淀为后续路线决策

---

## 9. 工件与目录约定

### 9.1 Configs

- `configs/stage6/`

用途：

- 存放所有 Stage 6 debug configs
- 不覆盖 Stage 5 主线 configs

### 9.2 Outputs

- `outputs/stage6/<family>/<run_name>/`

推荐 family：

- `metric_audit`
- `diagnostics`
- `tinyset`
- `smallsubset`
- `scale_fix`
- `depth_debug`
- `physics_audit`
- `summary`

### 9.3 Docs

- `docs/plan/stage_plan/stage6/`
- `docs/execution/`

原则：

- 规划与 issue 拆解放在 `docs/plan/stage_plan/stage6/`
- 真实运行报告与调试结论放在 `docs/execution/`

---

## 10. Stage 6 的最小完成标准

Stage 6 最低完成标准不是 “把所有 ablation 跑完”，而是：

1. eval / metric 语义已经被审计清楚。
2. raw intensity collapse 与 normalized reconstruction failure 已被显式拆分记录。
3. 至少一组 tiny-set 或 small-subset 结果能回答当前失败更像是：
   - loss identifiability
   - optimization failure
   - depth-specific instability
   - 或 physics-sensitive config 偏差
4. 至少一组 scale-fixing intervention 已完成，并给出结论：
   - 只修复 A
   - 同时改善 A 与 B
   - 或几乎无效
5. 文档中已明确说明是否允许恢复更宽的 Stage 6 systematic experiments。

更强完成标准：

1. 已明确判断 `L=5` 失败是否显著强于 `L=3`。
2. 已确认是否需要重开 distance mapping / phase range / crop 等 physics 假设。
3. Stage 6 已把后续工作收敛成少数几个高价值方向，而不是继续盲扫。

---

## 11. 一句话结论

Stage 6 的正确打开方式不是 “继续做更大的实验矩阵”，而是：

先把 `scale ambiguity / eval semantics / normalized reconstruction failure` 这三个层面的语义理清，再用高信息密度的小实验逐步定位主失效模式，最后才决定是否恢复系统化实验。
