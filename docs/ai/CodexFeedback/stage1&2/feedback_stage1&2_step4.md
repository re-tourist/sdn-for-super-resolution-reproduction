1. Task understanding  
实现 Stage1&2-4：新增一个**纯电子 bottleneck baseline**（不是通用 SR 大网络），语义严格对齐目标链路：  
`HR -> electronic encoder -> low-res latent -> electronic decoder -> reconstructed HR`  
并提供最小可运行训练脚本，使用现有统一评估入口 `evaluate_batch(...)` 输出 PSNR/SSIM，不改光学模块/插值 baseline/eval 实现。

2. Files added/modified  
- 新增 [electronic_baseline.py](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\models\electronic_baseline.py)  
- 新增 [train_electronic_baseline.py](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\scripts\train_electronic_baseline.py)  
- 未改你明确禁止的模块；`docs/execution/design_notes.md` 是你的已有改动，我已忽略。

3. Model structure  
模型类：`ElectronicBottleneckAutoencoder`  
- 输入默认：`[B,1,96,96]`  
- Encoder：两次 stride=2 下采样，`96 -> 48 -> 24`，最后 `1x1 conv` 投影到 latent  
- 显式瓶颈：默认 `latent_channels=1`，latent 形状默认 ` [B,1,24,24]`  
- Decoder：`24 -> 48 -> 96` 双上采样重建  
- 输出映射：`sigmoid`，显式约束到 `[0,1]`  
- 无 HR skip/bypass 跳连（严格经过 bottleneck）  
- `forward(..., return_latent=True)` 可直接拿到 latent，满足可检查性

4. Training/data assumptions  
训练脚本：`scripts/train_electronic_baseline.py`  
- Loss：仅 `L1/MAE`  
- Optimizer：Adam（默认 `lr=1e-3`）  
- 评估：复用 `from src.eval.evaluator import evaluate_batch`（未重写 PSNR/SSIM）  
- 数据：复用当前 Stage1/2 的最小 EMNIST 构造风格（与插值 baseline一致）  
  - 从 `EMNIST(train)` 抽样 `num_samples`  
  - 按 `val_ratio` 划分 train/val  
  - 每个 batch 将图像 resize 到 `hr_size`（默认 96）作为 HR  
- 默认训练配置：`epochs=10, batch_size=32, lr=1e-3`（已实现）

5. Run commands  
- Smoke test（我已实跑通过）：
```bash
python scripts/train_electronic_baseline.py \
  --output-dir outputs/electronic_baseline/smoke_run2 \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-3 \
  --seed 42 \
  --hr-size 96 \
  --sr-factor 4 \
  --latent-channels 1 \
  --num-samples 128 \
  --dataset-root outputs/cache/emnist \
  --no-download \
  --save-samples
```

- Normal training（建议）：
```bash
python scripts/train_electronic_baseline.py \
  --output-dir outputs/electronic_baseline/run_e10_b32 \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42 \
  --hr-size 96 \
  --sr-factor 4 \
  --latent-channels 1 \
  --num-samples 2000 \
  --dataset-root outputs/cache/emnist \
  --no-download \
  --save-samples
```

6. Output files  
`outputs/electronic_baseline/<run_name>/` 下会生成：
- `metrics.json`（best/final loss + PSNR/SSIM）
- `summary.json`（参数、数据构造、history、artifact路径）
- `checkpoints/best.pt`
- `checkpoints/final.pt`
- `samples/epoch_XXX.png`（开启 `--save-samples` 时）

7. Risks / known limitations  
- 当前实现只支持 `sr_factor=4`（CLI 有参数，但脚本会显式校验）  
- 数据仍是 Stage1/2 的最小 shell（EMNIST + resize），不是最终 paper 对齐的数据协议  
- val 是从 train 子集切分，不是独立官方验证集  
- 1 通道默认配置已对齐要求；若改多通道需同步检查数据与可视化逻辑

8. Completion status against acceptance criteria  
1) 可训练纯电子 baseline：满足  
2) 显式低分辨率 bottleneck：满足  
3) 默认 latent 为低分辨率单通道：满足（`[B,1,24,24]`）  
4) 无 HR skip bypass：满足  
5) 训练脚本可独立运行：满足（已 smoke 跑通）  
6) 输出重建样例、loss、PSNR、SSIM：满足  
7) 复用 `evaluate_batch(...)`：满足  
8) 包含 `metrics.json` 与 `summary.json`：满足  
9) 改动范围小：满足（仅新增 2 文件）  
10) 结构、数据假设、命令、限制已明确说明：满足