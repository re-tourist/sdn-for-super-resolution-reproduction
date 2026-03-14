# Troubleshooting

记录错误、排查过程、根因、修复方案和修复后的验证结果，避免同类问题重复出现。

## 问题记录模板

```md
## Issue <ID> - <问题标题>

### 基本信息
- 日期:
- 所属阶段:
- 相关模块:

### 问题描述
简要说明出现的异常。

### 现象
- 现象 1
- 现象 2

### 可能原因
- 原因 1
- 原因 2

### 排查过程
1. 步骤 1
2. 步骤 2

### 根本原因
最终确认的问题来源。

### 解决方案
- 修改代码:
- 修改配置:
- 验证方式:

### 修复后结果
- 结果 1
- 结果 2

### 经验总结
- 总结 1
- 总结 2
```

## Issue T001 - `configs/base.yaml` 覆盖显式 CLI 参数

### 基本信息
- 日期: 2026-03-14
- 所属阶段: Stage 1 / Stage 2
- 相关模块: `scripts/train_electronic_baseline.py`, `docs/run_order.md`, `configs/base.yaml`

### 问题描述
运行 `docs/run_order.md` 中的电子 baseline 命令时，训练实际参数与命令行不一致，导致实验配置、训练速度和结果偏离预期。

### 现象
- 在命令行显式传入 `--epochs 10 --batch-size 32 --lr 1e-3` 后，`summary.json` 中仍记录为 `epochs=50`, `batch_size=8`, `lr=1e-4`
- 服务器训练速度与预期不符
- `smallset_e10` 旧实验中 history 从 epoch 1 后几乎不变化，容易误判为模型本身完全失效
- `fit_one_sample` 旧报告中也出现了 `lr=1e-4`, `num_workers=4` 等非预期配置

### 可能原因
- 训练脚本读取了 `configs/base.yaml`
- CLI 参数没有真正覆盖 config 参数
- 参数解析逻辑把“显式传了默认值”误判成“没有传”

### 排查过程
1. 对比 `docs/run_order.md` 中命令与 `outputs/electronic_baseline/*/summary.json` 的 `train_args`
2. 发现命令写的是 `10 / 32 / 1e-3`，summary 实际记录为 `50 / 8 / 1e-4`
3. 检查 `configs/base.yaml`，确认其中正好包含 `epochs=50`, `batch_size=8`, `lr=1e-4`
4. 检查 `scripts/train_electronic_baseline.py`，确认旧实现用“是否等于 parser 默认值”判断 CLI 是否显式传参
5. 复现实验，确认只要 CLI 传入值与脚本默认值相同，就仍会被 config 覆盖

### 根本原因
旧脚本不是判断“用户是否显式传参”，而是判断“当前值是否等于 parser 默认值”。  
因此当用户显式传入的值恰好等于默认值时，脚本会错误地认为该参数未提供，并继续用 config 覆盖。

### 解决方案
- 修改代码:
  - 将关键 CLI 参数默认值统一设为 `None`
  - 增加 `choose_runtime_value(...)`，统一执行 `CLI > config > hardcoded default`
  - 保留 `--config configs/base.yaml` 用法，但不允许其覆盖显式 CLI
- 修改配置:
  - 不修改 `configs/base.yaml`
  - `docs/run_order.md` 中保留显式训练参数，作为实验可追踪依据
- 验证方式:
  - 本地重新运行训练脚本
  - 检查新的 `summary.json` 中 `train_args` 是否与命令行完全一致

### 修复后结果
- `fit_one_sample/summary.json` 记录为:
  - `lr = 0.001`
  - `num_samples = 1`
  - `overfit_single_sample = true`
- `smallset_e10/summary.json` 记录为:
  - `epochs = 10`
  - `batch_size = 32`
  - `lr = 0.001`
- 当前可以确认 CLI 参数覆盖问题已解决，训练结果和命令配置一致

### 经验总结
- 配置优先级必须显式设计，不能依赖“值是否等于默认值”这种脆弱判断
- 所有训练脚本都应把最终生效参数完整写入 `summary.json`
- 当实验结果异常时，先核对 `summary.json`，再判断模型是否有问题

## Issue T002 - `ReLU + sigmoid` 组合导致电子 baseline 早期塌缩为全黑输出

### 基本信息
- 日期: 2026-03-14
- 所属阶段: Stage 1 / Stage 2
- 相关模块: `src/models/electronic_baseline.py`, `scripts/train_electronic_baseline.py`, `outputs/electronic_baseline/*`

### 问题描述
修正参数覆盖问题后，电子 baseline 仍然训练异常。用户提供的重建图显示 `Recon` 一列几乎全黑，`difference map` 则保留了字符轮廓，说明模型输出退化为接近零图像。

### 现象
- `fit_one_sample` 虽然已经是正确的单样本 overfit 设定，但旧结果最终 `train_l1` 和 `val_l1` 仍停在约 `0.177`
- `fit_one_sample` 的旧 `final_val_psnr` 只有约 `8.56`
- `smallset_e10` 旧结果在第 1 个 epoch 后几乎完全停滞，`best_epoch=1`
- 用户提供的样图中 `Recon` 一列近乎全黑，而 `Latent` 仍保留了字符结构

### 可能原因
- `sigmoid` 输出层在稀疏灰度任务上过早进入饱和区
- `ReLU` 导致中间特征被大量截断，进一步恶化梯度流
- 白字黑底图像背景像素占多数，`L1` 优化容易先把整体输出压向黑色

### 排查过程
1. 确认这不是可视化保存 bug，因为用户提供的图和 JSON 同时表明模型确实输出接近零
2. 在本地用单样本直接训练旧模型，逐步打印 `recon mean/max/min` 和 `out_conv` 梯度
3. 观察到旧模型在约 10-15 步内把 `recon mean` 从约 `0.50` 压到接近 `0`
4. 到约第 30 步时，`recon max` 已接近 `0`，`out_conv` 梯度也衰减为 `0`
5. 这证明旧结构不是“学不会字符细节”，而是先塌缩到全黑，再因为 `sigmoid` 饱和失去恢复能力
6. 进一步做原型实验，验证将隐藏层改为 `LeakyReLU`、输出改为 `atan` 后，单样本 overfit 可恢复正常

### 根本原因
在 EMNIST 这类“黑背景 + 稀疏白前景”的任务上，旧模型使用的 `ReLU + sigmoid` 组合存在严重优化风险。

- `L1` 对背景像素的累计梯度占主导
- 优化初期最容易降低 loss 的方向是“把整体输出都压黑”
- 一旦输出 logits 被压得很负，`sigmoid` 迅速饱和
- `ReLU` 又进一步削弱中间层梯度流
- 最终网络进入“全黑输出 + 几乎无梯度”的坏解

### 解决方案
- 修改代码:
  - 将 `ConvBlock` 和 encoder 下采样层中的 `ReLU` 替换为 `LeakyReLU(0.1)`
  - 将输出映射从 `torch.sigmoid(out)` 替换为 `torch.atan(out) / torch.pi + 0.5`
  - 保持输出范围仍然显式约束在 `[0,1]`
- 修改配置:
  - 不需要新增配置项
  - 继续沿用当前 `run_order.md` 中的 overfit 和 smallset 命令
- 验证方式:
  - 本地单样本 overfit 40 epoch，确认不再塌缩
  - 服务器重新运行正式 overfit / smallset 命令并回传 JSON

### 修复后结果
- 本地单样本 overfit 40 epoch:
  - `best_val_psnr = 22.51`
  - `best_val_ssim = 0.8217`
  - `final_val_l1 = 0.0333`
- 服务器正式 `fit_one_sample`:
  - `best_val_psnr = 32.4633`
  - `best_val_ssim = 0.9624`
  - `final_val_psnr = 30.4862`
  - `final_val_ssim = 0.9619`
- 服务器正式 `smallset_e10`:
  - `final_val_l1 = 0.01042`
  - `final_val_psnr = 33.6274`
  - `final_val_ssim = 0.9780`
- 当前已经没有“Recon 全黑”或“epoch 1 后立即停滞”的现象

### 经验总结
- 对稀疏灰度重建任务，输出层激活函数的选择会直接影响是否出现早期梯度消失
- “输出全黑”不是简单的可视化问题，而是可以通过 tensor 统计和梯度统计确认的真实塌缩
- 在 baseline 阶段，单样本 overfit 是发现训练动力学问题的最快手段之一
- 对这类任务，除了看 `loss`，还应同时检查:
  - `recon mean/max/min`
  - 输出是否出现常值塌缩
  - preview 图是否与数值指标一致
