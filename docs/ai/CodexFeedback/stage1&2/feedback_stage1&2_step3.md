1. 任务理解  
你这次指定的是 `Stage1&2-2`：实现一个独立的 **interpolation baseline** 脚本（至少支持 `bilinear/bicubic`），流程为 `HR -> downsample -> LR -> interpolate -> SR`，并且**必须复用**已有统一评估入口 `evaluate_batch(...)` 来输出 `PSNR/SSIM`，同时保存可视化结果与 `metrics.json`。  
我按“最小改动”执行：只新增 1 个脚本，不改模型/训练/光学/现有 eval 文件。

2. 计划修改的文件  
- 新增 [run_interpolation_baseline.py](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\scripts\run_interpolation_baseline.py)

3. 实现说明  
- 新增脚本支持参数（按提示词要求）：
  - `--split train|val|test`
  - `--num-samples`
  - `--seed`
  - `--output-dir`
  - `--mode bilinear|bicubic`
  - `--save-grid / --no-save-grid`
  - `--save-individual / --no-save-individual`
  - `--download`
- 数据流程：
  - 从 EMNIST 读取样本（train/val/test；val 由 train 按 `val_ratio` 划分）
  - `HR`（默认会 resize 到 `--hr-size`，默认 96）
  - `LR`：`HR` 通过 bicubic 下采样到 `hr_size/scale`
  - `Pred`：`LR` 用 `--mode`（bilinear/bicubic）上采样回 HR 尺寸
- 评估复用：
  - 直接调用 `from src.eval.evaluator import evaluate_batch`
  - `preds` 和 `targets` 堆叠后统一计算 `PSNR/SSIM`
- 输出：
  - `metrics.json`（核心指标）
  - `summary.json`（运行参数 + 每样本信息）
  - 单样本图（可选）包含 `HR/LR/Interp/|Pred-HR|`
  - 总览图 `interpolation_grid.png`（可选）
- 已做实跑验证：
  - `bicubic` 和 `bilinear` 两种模式都可独立运行
  - 自动创建输出目录并写入指标与图像

4. 如何运行  
- 推荐（与提示词一致）：
```bash
python scripts/run_interpolation_baseline.py --split train --num-samples 32 --seed 42 --output-dir outputs/interpolation/train --mode bicubic
```
- bilinear 示例：
```bash
python scripts/run_interpolation_baseline.py --split val --num-samples 32 --seed 42 --output-dir outputs/interpolation/val --mode bilinear --save-grid --no-save-individual
```
- 我本地验证命令（使用已下载数据）：
```bash
python scripts/run_interpolation_baseline.py --split train --num-samples 4 --seed 42 --output-dir outputs/interpolation/train_smoke4 --mode bicubic --dataset-root outputs/cache/emnist --no-download
python scripts/run_interpolation_baseline.py --split val --num-samples 2 --seed 42 --output-dir outputs/interpolation/val_smoke2 --mode bilinear --dataset-root outputs/cache/emnist --no-download --save-grid --no-save-individual
```

5. 输出示例说明  
以 `outputs/interpolation/train_smoke4/` 为例，生成了：
- `metrics.json`
- `summary.json`
- `interpolation_grid.png`
- `sample_001_idx_....png` 等单样本图（开启 `--save-individual` 时）

`metrics.json` 示例字段：
- `mode`
- `split`
- `num_samples`
- `seed`
- `hr_size`
- `scale`
- `lr_size`
- `psnr_mean`
- `ssim_mean`

6. 风险与待确认点  
- 当前我默认 `hr_size=96, scale=4`（与当前 inspection 脚本保持一致）；如果你想严格按“stage shell 28x28”，可直接传 `--hr-size 28 --scale 4`（或你指定的组合）。  
- bicubic/bilinear 插值可能出现轻微数值越界（尤其 bicubic），评估模块内部会统一到 `[0,1]` 口径。  
- `val` 是从 EMNIST train 里按 `seed + val_ratio` 划分，不是官方单独 split。  

7. 完成状态（是否满足验收标准）  
满足。  
- 脚本可独立运行：是  
- 支持 bilinear / bicubic：是  
- 能从 HR 构造 LR 并回采样到 HR：是  
- 正确调用 `evaluate_batch(...)`：是  
- 输出 PSNR/SSIM：是  
- 保存可视化（单图/总览至少一种）：是  
- 输出目录自动创建：是  
- 未修改训练/模型/光学主流程：是  
- 改动范围小且清晰：是（仅新增 1 个脚本）  
- 对阶段数据设定与限制有明确说明：是