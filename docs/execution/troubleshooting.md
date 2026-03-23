# Troubleshooting

记录已确认的问题、排查过程、根因判断、修复状态与经验总结，避免同类问题重复出现。

## 问题记录模板

```md
## Issue <ID> - <问题标题>

### 基本信息
- 日期:
- 所属阶段:
- 相关模块:
- 当前状态: `open` / `mitigated` / `resolved`

### 问题描述
一句话说明问题。

### 现象
- 现象 1
- 现象 2

### 排查过程
1. 步骤 1
2. 步骤 2

### 根因判断
- 已确认根因:
- 尚未完全确认的部分:

### 影响范围
- 影响 1
- 影响 2

### 处理方案
- 已完成:
- 待完成:

### 验证结果
- 结果 1
- 结果 2

### 经验总结
- 总结 1
- 总结 2
```

---

## Issue T001 - `configs/base.yaml` 错误覆盖显式 CLI 参数

### 基本信息
- 日期: 2026-03-14
- 所属阶段: Stage 1 / Stage 2
- 相关模块: `scripts/train_electronic_baseline.py`, `configs/base.yaml`, `docs/run_order.md`
- 当前状态: `resolved`

### 问题描述
训练脚本在读取 `configs/base.yaml` 后，会错误覆盖用户在命令行里显式传入的训练参数，导致实际运行配置与命令不一致。

### 现象
- 明明传入了 `--epochs 10 --batch-size 32 --lr 1e-3`，`summary.json` 仍记录为旧配置值。
- 训练速度、loss 变化和预期不一致。
- 早期实验容易被误判为“模型无效”，实际是配置没按命令生效。

### 排查过程
1. 对比 `docs/run_order.md` 中命令与 `outputs/electronic_baseline/*/summary.json` 中的 `train_args`。
2. 发现 summary 记录的值与 `configs/base.yaml` 一致，而不是与 CLI 一致。
3. 检查脚本参数合并逻辑，确认旧实现通过“是否等于 parser 默认值”来判断用户是否显式传参。

### 根因判断
- 已确认根因:
  - 配置优先级实现错误。
  - 旧逻辑无法可靠判断“用户是否显式传参”。
- 尚未完全确认的部分:
  - 无。

### 影响范围
- Stage 1 / Stage 2 的电子 baseline 结果可追踪性。
- 训练配置、速度、指标与实验记录的一致性。

### 处理方案
- 已完成:
  - 关键 CLI 参数默认值改为 `None`。
  - 统一为 `CLI > config > hardcoded default`。
  - 保留 `--config` 用法，但不再允许其覆盖显式 CLI。
- 待完成:
  - 无。

### 验证结果
- `fit_one_sample` 与 `smallset_e10` 的 summary 已与命令参数一致。
- 当前可以信任 `summary.json` 中记录的最终运行参数。

### 经验总结
- 配置优先级必须显式设计，不能依赖脆弱的“是否等于默认值”判断。
- 当训练结果异常时，先核对 `summary.json`，再判断模型本身是否有问题。

---

## Issue T002 - `ReLU + sigmoid` 组合导致电子 baseline 早期塌缩为全黑输出

### 基本信息
- 日期: 2026-03-14
- 所属阶段: Stage 1 / Stage 2
- 相关模块: `src/models/electronic_baseline.py`, `scripts/train_electronic_baseline.py`
- 当前状态: `resolved`

### 问题描述
旧电子 baseline 在 EMNIST 这类稀疏白字黑底任务上，会很快塌缩到近乎全黑输出，导致 overfit 和 small-set 训练都明显异常。

### 现象
- `Recon` 图几乎全黑，但 `difference map` 保留字符轮廓。
- 单样本 overfit 的 `PSNR / SSIM` 明显偏低。
- small-set 训练在很早期就停滞。

### 排查过程
1. 确认不是可视化 bug，而是模型输出本身接近 0。
2. 本地打印 `recon mean/max/min` 和输出层梯度，确认输出层快速进入饱和区。
3. 对比激活函数与输出映射后，确认 `ReLU + sigmoid` 对该类稀疏任务不稳定。

### 根因判断
- 已确认根因:
  - 在黑底稀疏目标上，`L1` 目标容易先把整体输出压向黑色。
  - 一旦输出 logits 被过度压低，`sigmoid` 很快饱和。
  - `ReLU` 又进一步削弱中间层梯度流。
- 尚未完全确认的部分:
  - 无。

### 影响范围
- 电子 baseline 的单样本过拟合能力。
- 小子集训练的稳定性与结果可信度。

### 处理方案
- 已完成:
  - 隐藏层激活从 `ReLU` 改为 `LeakyReLU(0.1)`。
  - 输出映射从 `sigmoid` 改为有界 `atan` 映射。
- 待完成:
  - 无。

### 验证结果
- 单样本 overfit 恢复正常。
- small-set 训练不再快速塌缩为全黑输出。
- baseline 指标显著改善。

### 经验总结
- “全黑输出”往往不是简单的可视化问题，而是真实训练动力学问题。
- 在稀疏灰度重建任务里，输出层映射和激活函数选择会直接决定是否发生早期塌缩。

---

## Issue T003 - Stage 5 `L=5` 主运行出现原始输出强度塌缩，`preview_final.png` 中 `pred_roi` 近乎全黑

### 基本信息
- 日期: 2026-03-20
- 所属阶段: Stage 5 / Stage 6 handoff
- 相关模块:
  - `src/losses/stage5_sr_loss.py`
  - `scripts/train_stage5_paper.py`
  - `docs/execution/stage5_main_run_report.md`
  - `outputs/stage5/main_run/stage5_l5_phase_main/*`
  - `outputs/stage5/eval/stage5_l5_phase_main_*/summary.json`
  - `outputs/stage5/blind_eval/stage5_l5_phase_main_blind_eval/summary.json`
- 当前状态: `open`

### 问题描述
Stage 5 主运行虽然完成了完整的 paper-aligned `phase-only / L=5` 训练、regular eval 和 blind eval，但训练后的原始 `I_out_roi` 强度极低，导致 `preview_final.png` 中的 `pred_roi` 近乎全黑，且最终 `PSNR / SSIM` 明显落后于 bicubic baseline。

### 现象
- `outputs/stage5/main_run/stage5_l5_phase_main/preview_final.png` 中：
  - `x_hr` 与 `target_roi` 清晰可见；
  - `pred_roi` 在固定 `[0, 1]` 显示范围下近乎全黑；
  - `abs_error` 基本复现了 target 轮廓。
- `docs/execution/stage5_main_run_report.md` 记录的 regular eval：
  - val:
    - model `PSNR = 14.9181`
    - model `SSIM = 0.8126`
    - bicubic `PSNR = 25.6875`
    - bicubic `SSIM = 0.9448`
  - test:
    - model `PSNR = 9.5774`
    - model `SSIM = 0.4386`
    - bicubic `PSNR = 20.2487`
    - bicubic `SSIM = 0.8393`
- blind eval `per_target_records` 中，多数 `prediction_sum` 远小于对应 target 的能量总和。
- 本地对主运行 best checkpoint 的抽样复核显示：
  - raw `pred_roi.max ≈ 0.0064`
  - raw `pred_roi.mean ≈ 1.95e-4`
  - 每样本 `sigma ≈ 177 ~ 212`

### 排查过程
1. 先确认这不是简单的图片保存或 colormap bug。
   - `scripts/train_stage5_paper.py` 的 `save_preview_grid(...)` 确实直接绘制 raw `pred_roi`，并固定使用 `vmin=0, vmax=1`。
2. 再检查 loss 语义。
   - `src/losses/stage5_sr_loss.py` 的 Stage 5 主项是：
     - `mean(|y - sigma * y_hat|)`
   - 其中
     - `sigma = sum(y) / (sum(y_hat) + epsilon)`
3. 检查 `L=5` 的效率项配置。
   - 当前冻结配置是 `gamma_by_depth[5] = 0.0`。
   - 这意味着 `L=5` 主线训练没有额外的输出能量约束。
4. 本地加载 `outputs/stage5/main_run/stage5_l5_phase_main/checkpoints/checkpoint_best.pt`，对 `val` 子集做复核。
   - raw 预测极暗，但不是严格全零。
   - `sigma * pred_roi` 的均值可与 target 对齐。
5. 进一步比较 raw 与 `sigma` 重标定后的指标。
   - 在 `val16` 子集上：
     - raw `PSNR ≈ 14.79`, `SSIM ≈ 0.811`
     - rescaled `PSNR ≈ 14.78`, `SSIM ≈ 0.811`
   - 在 `test16` 子集上：
     - raw `PSNR ≈ 9.85`, `SSIM ≈ 0.470`
     - rescaled `PSNR ≈ 9.84`, `SSIM ≈ 0.470`
6. 结论是：
   - “pred_roi 发黑”确实首先来自原始强度过低；
   - 但问题不只是可视化动态范围太小，模型本身的经验效果也确实不好。

### 根因判断
- 已确认根因:
  - 当前 `L=5` 的主损失存在显著的全局尺度不确定性：
    - 使用 `sigma * y_hat` 与 target 比较；
    - 且 `gamma_by_depth[5] = 0.0` 时，没有输出能量惩罚项；
    - 因此模型可以把 raw `I_out_roi` 压得很低，而仍在 normalized loss 下获得不坏的数值。
  - `preview_final.png` 画的是 raw `pred_roi`，所以这一退化解会被直观地显示为“全黑”。
- 尚未完全确认的部分:
  - 这是否是当前主结果不佳的唯一主因。
  - 仍需进一步区分：
    - 是纯粹的 scale collapse；
    - 还是同时叠加了 geometry / phase range / distance mapping / optimization stability 等更深层问题。

### 影响范围
- 影响 Stage 5 主运行结果的物理解读：
  - 当前不能把主运行结果解释为“已成功恢复高频结构”。
- 影响 preview 可解释性：
  - 仅看 raw `pred_roi` 容易把问题理解成“完全无输出”。
- 不影响以下结论：
  - Stage 5 工程链路已经真实落地；
  - 主运行、regular eval、blind eval 工件都是真实存在且可复核的。

### 处理方案
- 已完成:
  - 确认 `preview_final.png` 的黑图不是文件损坏，而是 raw 输出强度极低导致的真实现象。
  - 确认 Stage 5 主运行当前应被解读为：
    - 工程闭环成立；
    - 经验结果质量不理想；
    - 不能宣称论文效果复现成功。
- 待完成:
  - 在训练可视化中同时保存：
    - raw `pred_roi`
    - `sigma * pred_roi`
    - 必要时再加每样本归一化版本
  - 在 summary / report 中显式记录：
    - raw output sum
    - target sum
    - sigma 统计
  - 进入 Stage 6 做定向排错，而不是继续把当前主运行当成成功复现结果。

### 验证结果
- 本地复核确认：
  - raw `pred_roi` 不是严格全零；
  - 但绝对强度确实非常低；
  - 该现象与 main-run report、regular eval、blind eval 的差结果是一致的。
- 当前可成立的结论是：
  - `Stage 5 engineering PASS`
  - `empirical reproduction NOT achieved`

### 经验总结
- 对带有 `sigma` 重标定的 loss，必须区分：
  - raw 输出是否有物理意义；
  - 重标定后 loss 是否下降。
- “preview 黑图”既可能是显示范围问题，也可能是真实的低能量退化；必须结合 tensor 统计、loss 语义和 eval 结果一起判断。
- 后续 Stage 6 的第一优先级不应是大而全 sweep，而应是围绕 `L=5` 的强度塌缩 / scale ambiguity 做定向排错。

