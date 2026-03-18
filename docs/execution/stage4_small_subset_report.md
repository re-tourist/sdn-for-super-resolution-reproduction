# Stage 4 Small-Subset Report

## 1. 任务边界

- 本次只验证 Stage 4 小子集 closed-loop learnability
- 不重写 optical core
- 不重写 dataset / adapter / wrapper / trainer
- 不进入 Stage 5 paper-aligned settings

## 2. 运行命令

实际运行（根据 `config_snapshot.json` 还原的参数）：

```bash
python scripts/train_stage4_minimal.py \
  --subset-size 16 \
  --batch-size 4 \
  --steps 200 \
  --preview-limit 4 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_7_small_subset_16_s200
```

补充运行（更强验证，仍基于同一子集与配置）：

```bash
python scripts/train_stage4_minimal.py \
  --subset-size 16 \
  --batch-size 4 \
  --steps 400 \
  --preview-limit 4 \
  --device cuda \
  --dataset-root data/raw/emnist \
  --run-name issue4_7_small_subset_16_s400
```

## 3. 设备与工件目录

- 设备：`cuda`
- 工件目录：
  - `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s200/`
  - `outputs/stage4/minimal_trainer/issue4_7_small_subset_16_s400/`

## 4. Core Scalar Results

### 200 steps

- initial loss: `0.217708`
- best loss: `0.149736` @ step 171
- final loss: `0.180445`
- val loss: `0.192186`
- val PSNR / SSIM: `8.3882` / `0.0557`

### 400 steps

- initial loss: `0.217708`
- best loss: `0.144116` @ step 360
- final loss: `0.171176`
- val loss: `0.190612`
- val PSNR / SSIM: `8.3843` / `0.0556`

### 趋势结论

- train loss 明确下降，但波动较大（小子集 + mini-batch 导致）
- val loss 从 `0.2286` 降到约 `0.19` 后趋于平台
- 与 single-sample 不同，小子集 loss 不会接近 0，是可预期现象

## 5. Preview / Artifact Inspection Findings

- 多个输入是否不同：是（预览行对应不同 EMNIST 字符）
- 多个输出是否随输入变化：是，但变化主要体现在亮斑位置 / 形状的偏移，细节与 `target_roi` 仍不匹配
- 是否出现明显 collapse 到单一亮斑 / 单一模式：否，输出仍受输入影响，但整体偏“模糊亮斑”

## 6. Gradient Findings

- encoder gradients：全程观测到非零梯度（200/200、400/400 steps）
- optics gradients：全程观测到非零梯度（200/200、400/400 steps）

## 7. Stability Findings

- NaN / Inf：未发现（`run_summary.json` 显示 `initial_preview_finite=true`、`final_preview_finite=true`）
- 是否存在明显数值不稳定：未见明显不稳定或爆炸

## 8. Verdict

**PASS（带保留项）**

理由：

- loss 在小子集上有明确下降，且无数值异常
- encoder / optics 梯度持续可观测
- 输出对输入有响应，没有明显 collapse

保留项：

- 预测仍以模糊亮斑为主，细节不足，val PSNR/SSIM 偏低
- 说明最小闭环“可学”成立，但距离 paper-final 读出质量仍有差距

## 9. 非阻塞未决项

- loss 平台值较高，可能与最小 encoder 容量、toy optics grid、normalized MAE 的 per-sample 归一化有关
- 需要在 Stage 4 总结中明确为“可学习性成立、质量尚不达标”

## 10. 下一步

进入 Stage 4 总结与 Stage 5 GO/NO-GO 决策整理，明确：

- 单样本与小子集均通过可学习性验收
- 小子集仍存在输出模糊与指标偏低的现象（非阻塞）
