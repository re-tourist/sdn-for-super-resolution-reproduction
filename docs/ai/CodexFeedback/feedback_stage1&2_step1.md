1. 任务理解  
已按 Stage 1/2 的最小改动原则，实现了一个独立的数据抽样可视化脚本，用于检查 `HR -> downsample -> LR -> upsample` 链路，不修改模型、训练、评估主流程。

2. 计划修改的文件  
- 新增: [scripts/inspect_dataset.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/scripts/inspect_dataset.py)

3. 实现说明  
- 复用现有工程配置入口：脚本会读取 `configs/base.yaml`（若可读）中的 `data.root`、`data.scale`，并在 `data.root` 不可用时回退到 `data/raw/emnist`。  
- 数据加载：使用 `torchvision.datasets.EMNIST`，支持 `--split train|val|test`。  
- `val` 切分：当前项目无现成 val 文件时，按 `--val-ratio` 从 EMNIST train 集合按 seed 可复现划分。  
- 抽样与可视化：  
  - 固定随机种子抽样（`--seed`）  
  - 每样本输出 `HR / LR / upsampled LR` 三列  
  - 标题包含 index、shape、min/max（灰度使用 `cmap='gray'`）  
- 输出控制：  
  - `--save-individual/--no-save-individual`  
  - `--save-grid/--no-save-grid`  
  - 自动创建输出目录  
- 运行摘要：输出 `summary.json`，记录 `split/num_samples/seed/output_dir/upsample_mode` 等参数与样本统计。  
- 已本地实测可运行（train/val/test 均通过）。

4. 如何运行  
- 你的建议接口可直接用：  
```bash
python scripts/inspect_dataset.py --split train --num-samples 8 --seed 42 --output-dir results/inspection/train --upsample bicubic
```
- 若本地还没有 EMNIST，可加下载参数：  
```bash
python scripts/inspect_dataset.py --split train --num-samples 8 --seed 42 --output-dir results/inspection/train --upsample bicubic --download
```
- 可选常用参数：  
`--emnist-split letters` `--val-ratio 0.1` `--fix-emnist-orientation` `--no-save-individual` `--no-save-grid`

5. 输出示例说明  
在 `--output-dir` 下会生成（按参数开关）：  
- `sample_001_idx_xxxxxx.png`（每样本三联图）  
- `inspection_grid.png`（总览网格图，每行一个样本，三列 HR/LR/Upsampled）  
- `summary.json`（本次参数与样本 shape/min/max 统计）

6. 风险与待确认点  
- 当前仓库没有统一 dataset/dataloader 代码；本脚本使用 `torchvision EMNIST` 做最小封装（未改训练入口）。  
- `val` 为脚本内按比例切分，不是项目固定官方 split 文件。  
- `configs/base.yaml` 当前 `data.root` 是占位风格，脚本做了回退；正式训练前建议统一数据根目录约定。  
- bicubic 插值可能出现轻微越界（<0 或 >1），已在图标题和 summary 中暴露 min/max，便于排查归一化问题。

7. 完成状态（是否满足验收标准）  
满足。  
- 脚本可独立运行  
- 可随机导出样本图  
- 每样本包含 HR / LR / upsampled LR  
- 支持固定 seed  
- 输出目录自动创建  
- 未改动训练/模型代码  
- 改动范围仅新增一个独立脚本