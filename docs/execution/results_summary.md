# Results Summary

阶段性结果总结，强调当前已确认的结论、可接受范围和后续工作重点。

## 当前阶段
- Stage 1 / Stage 2
- 目标: 完成纯电子 baseline、数据协议检查、统一评估协议和最小可训练闭环验证

## 已确认结果

### 1. 电子 baseline 已恢复正常训练
- 模型: `ElectronicBottleneckAutoencoder`
- 输入 / 输出: `[B,1,96,96] -> [B,1,24,24] latent -> [B,1,96,96]`
- bottleneck:
  - 空间尺寸: `24 x 24`
  - 通道数: `1`
- 评估入口: 复用 `src.eval.evaluator.evaluate_batch(...)`

### 2. 单样本 overfit sanity check 通过
来源: [fit_one_sample/metrics.json](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\outputs\electronic_baseline\fit_one_sample\metrics.json)

- `best_epoch = 194`
- `best_val_psnr = 32.4633`
- `best_val_ssim = 0.9624`
- `final_val_psnr = 30.4862`
- `final_val_ssim = 0.9619`

结论:
- 模型已经能够在相同 bottleneck 约束下记住单样本
- 这说明当前训练骨架、数据构造、loss 和输出映射已基本正确

### 3. 小数据集训练结果正常
来源: [smallset_e10/metrics.json](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\outputs\electronic_baseline\smallset_e10\metrics.json)

- 训练集 / 验证集:
  - `train_samples = 1800`
  - `val_samples = 200`
- 最终结果:
  - `final_val_l1 = 0.01042`
  - `final_val_psnr = 33.6274`
  - `final_val_ssim = 0.9780`
- history 趋势:
  - epoch 1: `PSNR = 24.12`, `SSIM = 0.8305`
  - epoch 5: `PSNR = 30.62`, `SSIM = 0.9619`
  - epoch 10: `PSNR = 33.63`, `SSIM = 0.9780`

结论:
- 训练过程平稳，指标持续提升
- 没有再出现早期塌缩、全黑重建或 epoch 1 后停滞

## 已解决问题

### 问题 1: config 覆盖 CLI
- 旧问题: `configs/base.yaml` 会覆盖命令行显式传入的训练参数
- 影响: 训练速度、学习率、batch size、epoch 数全部可能失真
- 当前状态: 已修复

### 问题 2: `ReLU + sigmoid` 导致全黑输出塌缩
- 旧问题: 稀疏灰度图像任务中，模型会快速塌缩到零输出并丢失梯度
- 当前修复:
  - 隐层激活改为 `LeakyReLU(0.1)`
  - 输出映射改为 `atan(out) / pi + 0.5`
- 当前状态: 已修复

## 当前判断
- 当前电子 baseline 结果正常
- 当前电子 baseline 可以通过 Stage 1 / Stage 2 验收
- 当前结果可以作为后续 optical model 的纯电子参考

## 风险与边界
- 当前结果仍是 Stage 1 / Stage 2 的最小 shell 数据设定，不代表已完成论文最终设定对齐
- 当前结论只说明“同一 bottleneck 约束下，纯电子方案可学且训练流程正常”
- 后续进入 optical model 时，仍需要重新检查:
  - optical bottleneck 对齐方式
  - 相同数据协议下的可学习性
  - 评估协议是否保持一致

## 下一步建议
1. 固定当前电子 baseline，作为后续对照组
2. 在不改变评估协议的前提下推进 optical decoder 主线
3. 后续所有光学实验都应至少和当前 `smallset_e10` 级别结果进行对比
