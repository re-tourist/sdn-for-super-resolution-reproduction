# Stage 4 Single-Sample Report

## 1. 任务边界

- 本次只验证 Stage 4 单样本 closed-loop learnability
- 不重写 optical core
- 不重写 dataset / adapter / wrapper / trainer
- 不进入 Stage 5 paper-aligned settings

## 2. 运行命令

正式验收运行：

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 100 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --download \
  --run-name issue4_6_single_sample_100
```

补充运行：

```bash
python scripts/train_stage4_minimal.py \
  --single-sample \
  --steps 300 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_6_single_sample_300
```

## 3. 设备与工件目录

- 设备：`cuda`
- 工件目录：
  - `outputs/stage4/minimal_trainer/issue4_6_single_sample_100/`
  - `outputs/stage4/minimal_trainer/issue4_6_single_sample_300/`

## 4. Core Scalar Results

| run | steps | initial loss | best loss | final loss | encoder grad | optics grad | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
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

## 5. Preview / Artifact Inspection Findings

- `preview_step0.png` 中的 `pred I_out_roi` 主要是扩散亮斑，和 `target_roi` 不对齐。
- 到 100-step 时，预测已从泛化亮斑收缩成更接近字符拓扑的结构。
- 到 300-step 时，预测主轮廓进一步贴近 target，背景更暗，误差主要集中在字符边缘与局部笔画残差。

说明：

- 当前预览图展示的是 raw `I_out_roi`。
- 训练目标是 ROI-level normalized MAE，内部存在按样本重标定的 `sigma`。
- 因此 raw 读出预览、PSNR/SSIM 与训练主损失不完全同向。

## 6. Gradient Findings

- 100-step：`encoder_nonzero_steps = 100 / 100`，`optics_nonzero_steps = 100 / 100`
- 300-step：`encoder_nonzero_steps = 300 / 300`，`optics_nonzero_steps = 300 / 300`

结论：

- encoder 与 optics 在整个训练过程中都持续收到有效梯度

## 7. Stability Findings

- `run_summary.json` 中 `initial_preview_finite = true`
- `run_summary.json` 中 `final_preview_finite = true`
- 无 NaN / Inf
- loss curve 连续，没有爆炸或异常跳变

## 8. Verdict

**PASS**

理由：

1. 单样本 closed-loop loss 明确下降。
2. 预测 ROI 从初始扩散亮斑演化到与目标更接近的结构。
3. encoder 与 optics 梯度在整个训练中均被持续观测到。
4. 没有 NaN / Inf 或明显数值不稳定。

## 9. 非阻塞保留项

- 300-step 的 raw PSNR / SSIM 低于 100-step，不应忽略。
- 在当前 Stage 4 最小协议下，这更像是 normalized-MAE 训练目标与 raw ROI 可视化指标不完全一致，而不是 learnability 消失。
- 当前 PASS 的含义是“单样本 learnability 已成立”，不是“paper-final 读出质量已经理想”。

## 10. 下一步

进入 Stage 4 的小子集闭环验收，重点检查：

- 小子集下 loss 是否仍能稳定下降
- 梯度可观测性是否保持
- 预览图是否从单样本记忆延伸到少量真实样本上的持续可学
