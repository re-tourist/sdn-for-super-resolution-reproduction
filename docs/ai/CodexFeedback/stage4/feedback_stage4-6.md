## Issue 4.6 Feedback

### 任务边界

本次只执行 Stage 4 的正式 single-sample overfit 验收，不改 optical core，不改 dataset / adapter / wrapper，不扩展成 Stage 5 或通用训练框架。分析对象仅限已有 trainer `scripts/train_stage4_minimal.py` 生成的单样本运行工件。

### 使用命令

首轮正式验收运行：

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 100 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --download \
  --run-name issue4_6_single_sample_100
```

由于首轮已经清楚下降，但为确认 learnability 趋势又补跑了更强的单样本版本：

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 300 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_6_single_sample_300
```

### 设备与工件目录

- 设备：`cuda`
- 本地回传工件目录：
  - `outputs/stage4/minimal_trainer/issue4_6_single_sample_100/`
  - `outputs/stage4/minimal_trainer/issue4_6_single_sample_300/`

两次运行都包含：

- `config_snapshot.json`
- `history.json`
- `run_summary.json`
- `loss_curve.png`
- `preview_step0.png`
- `preview_best.png`
- `preview_final.png`
- `phi_preview_step0.png`
- `phi_preview_best.png`
- `phi_preview_final.png`
- `grad_stats.json`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_latest.pt`

### 核心结果摘要

| run | steps | initial loss | best loss | final loss | encoder grad | optics grad | NaN/Inf |
|---|---:|---:|---:|---:|---|---|---|
| `issue4_6_single_sample_100` | 100 | 0.213564 | 0.164205 | 0.164152 | observed | observed | none |
| `issue4_6_single_sample_300` | 300 | 0.213564 | 0.117267 | 0.116939 | observed | observed | none |

补充指标：

- 100-step final val metrics:
  - `val_loss = 0.164152`
  - `val_psnr = 12.1184`
  - `val_ssim = 0.3732`
- 300-step final val metrics:
  - `val_loss = 0.116939`
  - `val_psnr = 10.8155`
  - `val_ssim = 0.2243`

### 工件检查结论

#### 1. Loss 是否清楚下降

是。

- 100-step 的 train / val loss 基本同步、平滑下降，末端仍在下降，没有数值爆炸或停滞。
- 300-step 继续把 normalized MAE 从 `0.213564` 压到 `0.116939`，说明当前最小闭环在单样本上存在持续可学习性。

#### 2. 输出 ROI 是否更接近 target

是，但要带上一个重要限定：

- `preview_step0.png` 中的 `pred I_out_roi` 主要是扩散的大块亮斑，和 `target_roi` 的字符结构不对齐。
- 到 100-step 时，预测已经从“泛化亮斑”收缩成可辨认的字符拓扑，`|pred-target|` 也从大面积高误差变成更集中在笔画附近。
- 到 300-step 时，预测的空间结构进一步贴近 target 的主轮廓，背景明显更暗，误差图主要集中在字符边缘和局部笔画残差。

限定项：

- 当前预览图展示的是 raw `I_out_roi`，而训练目标是 ROI-level normalized MAE，内部存在按样本重标定的 `sigma`。
- 因此 raw 读出预览、PSNR/SSIM 与训练主损失并不完全同向。300-step 上 loss 明显更低，但 raw `val_psnr/val_ssim` 反而低于 100-step，这更像是“训练目标与 raw 强度可视化不完全一致”，而不是 learnability 消失。

#### 3. Encoder 与 optics 梯度是否非零

是。

- 100-step：`encoder_nonzero_steps = 100 / 100`，`optics_nonzero_steps = 100 / 100`
- 300-step：`encoder_nonzero_steps = 300 / 300`，`optics_nonzero_steps = 300 / 300`

这说明当前单样本闭环训练中，encoder 和 optics 参数都持续收到了有效梯度。

#### 4. 是否出现 NaN / Inf 或明显数值异常

没有。

- `run_summary.json` 中 `initial_preview_finite = true`、`final_preview_finite = true`
- loss curve 连续，没有出现爆炸或异常跳变
- `phi_preview_final.png` 显示相位图已经形成结构化模式，并覆盖接近完整的相位范围，但没有数值崩溃迹象

### Verdict

**PASS**

理由：

1. 单样本 closed-loop loss 明确下降，且 100-step 已足以构成正式验收证据。
2. 预测 ROI 从初始扩散亮斑演化到与目标字符轮廓更接近的结构。
3. encoder 与 optics 梯度在整个训练过程中均被持续观测到。
4. 没有 NaN / Inf 或明显数值不稳定。

保留说明：

- 300-step 的 raw PSNR/SSIM 回落，不应被忽略，但在当前 Stage 4 最小协议下，它更像是“当前 normalized-MAE 训练目标与 raw ROI 可视化指标不完全一致”的现象，而不是 Issue 4.6 的验收失败。
- 因此本次 PASS 是“最小单样本 learnability 已成立”的 PASS，不是“paper-final 读出质量已经理想”的 PASS。

### 最可能的后续关注点

当前最窄的下一步，不是重写 trainer，也不是跳到 Stage 5，而是继续做下一个最小验收任务：

- 进入 Stage 4 的 **small-subset acceptance run**
- 继续使用同一 trainer、同一 dataset path、同一 wrapper
- 重点检查：
  - 小子集下 loss 是否仍能稳定下降
  - 梯度可观测性是否保持
  - 预览图是否从单样本记忆延伸到“少量真实样本上的持续可学”

如果后续需要更细地解释 300-step 的 PSNR/SSIM 回落，最窄的调试目标应是：

- 在分析侧额外区分 raw `I_out_roi` 与 loss 内部 `sigma` 重标定后的读出

这属于验收解释增强，不构成当前 Issue 4.6 的阻塞。
