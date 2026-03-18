4.3 的边界已按要求收住：这次只补了最小真实 EMNIST 数据路径和显式 `target_hr -> target_roi` adapter，用来服务后续 closed-loop training；没有实现 trainer、loss loop，也没有改 optical core 或 wrapper 核心职责。

变更文件：
- [src/datasets/stage4_emnist.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/datasets/stage4_emnist.py)
- [src/datasets/target_adapter.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/datasets/target_adapter.py)
- [src/datasets/__init__.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/datasets/__init__.py)

变更总结：
- 新增 `Stage4EMNISTDataset`，复用可信 EMNIST 路径，产出显式字段 `x_hr / target_hr / label / dataset_index`
- 新增 `build_stage4_train_val_datasets(...)`，沿用 baseline 风格做最小 train/val split 和 single-sample overfit 入口
- 新增 `Stage4RoiTargetAdapter`，把 `target_hr -> target_roi` 的 resize/clamp 逻辑从脚本里抽出来
- 新增 `save_stage4_input_target_preview(...)`，保存 `x_hr / target_roi / target_roi(upsampled)` 预览
- 没有修改现有 optical module、minimal phase encoder、minimal hybrid wrapper

Stage 4 当前 `x_hr / target_roi` 定义：
- `x_hr`：来自 `torchvision.datasets.EMNIST` 的真实单通道图像，经 bicubic resize 到当前 Stage 4 最小 HR 尺寸，当前 smoke 使用 `96x96`
- `x_hr` 形状：`[B, 1, H_hr, W_hr]`，当前验证为 `[B, 1, 96, 96]`
- `x_hr` 数值：`float32`，并在数据侧显式 clamp 到 `[0, 1]`
- `target_hr`：当前最小协议里与 `x_hr` 指向同一张真实 HR 图像内容，用于后续显式 target adaptation；这是 Stage 4 minimal learnability 选择，不是 paper-final protocol
- `target_roi`：由 `Stage4RoiTargetAdapter` 显式从 `target_hr` 生成，当前实现是 resize 到 optical `output_crop_hw` 后再 clamp 到 `[0, 1]`
- `target_roi` 形状：`[B, 1, H_roi, W_roi]`，并要求与 `I_out_roi` 完全一致；当前 smoke 为 `[B, 1, 20, 20]`

preview 保存位置与内容：
- 预览文件已保存到 [outputs/stage4/data_preview_smoke/pair_preview.png](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/outputs/stage4/data_preview_smoke/pair_preview.png)
- 内容包含三列：`x_hr`、`target_roi`、`target_roi (upsampled)`
- 我已实际查看这张图，字形可解释，没有明显空白或错位

batch / shape 自检结果：
- 已用真实 EMNIST batch 跑通最小 smoke check
- 数据构造：`num_samples=16`，得到 `train=12, val=4`
- batch 成功加载
- `x_hr.shape = (4, 1, 96, 96)`
- `target_hr.shape = (4, 1, 96, 96)`
- `target_roi.shape = (4, 1, 20, 20)`
- 现有 wrapper 在同批 `x_hr` 上 forward 成功，`I_out_roi.shape = (4, 1, 20, 20)`
- 已检查 `x_hr / target_hr / target_roi / phi_lr / I_out_roi` 无 `NaN/Inf`

下一步建议：
- 进入最小 Stage 4 训练脚本 issue
- 训练脚本直接复用这三个接口：
  - `batch["x_hr"]`
  - `Stage4RoiTargetAdapter(batch["target_hr"])`
  - `MinimalHybridWrapper(x_hr) -> I_out_roi`
- 先做单样本 overfit，再做小子集 closed-loop run，不要回头再改 wrapper