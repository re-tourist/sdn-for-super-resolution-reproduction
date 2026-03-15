本次已按 Stage 3 / Issue 5 的边界实现最小单样本拟合入口，没有引入 encoder，也没有改 decoder 接口或 propagation core。

修改的文件：
- 新增 [train_decoder_only_single_sample.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/train_decoder_only_single_sample.py)
- 更新 [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md):18

新增的可运行入口：
- `python scripts/train_decoder_only_single_sample.py --depth 3 --steps 150`
- 脚本默认做一张确定性的单目标 ROI 图像拟合，创建可学习输入 phase 参数，走当前 optical decoder 的 `I_out_roi` 输出，并保存：
  - `summary.json`
  - `artifacts.pt`
  - `roi_triptych.png`
  - `loss_curve.png`

使用的 loss / objective：
- 使用 Stage 3 冻结的 engineering default `normalized MAE`，只在 ROI 上计算
- 具体实现是：
  - `sigma = sum(target) / (sum(pred) + epsilon)`
  - 逐样本 ROI 归一化后计算 `mean(abs(target - sigma * pred))`
  - batch 再取平均
- 这次没有引入 PSNR/SSIM，也没有引入额外训练目标

单样本拟合是否实际出现了 loss 下降：
- 是，已实际运行验证
- 我运行的是：
  - `python scripts/train_decoder_only_single_sample.py --depth 3 --steps 120 --output-dir outputs/optics/decoder_only_single_sample_smoke`
- 实际结果：
  - `initial_loss = 0.1463066`
  - `best_loss = 0.0381246`
  - `final_loss = 0.0389387`
  - `loss_decrease = 0.1081820`
  - `roi_change_mean_abs = 0.4759185`
  - `best_roi_std = 0.3469959`
- 这说明：
  - loss 明显下降
  - `I_out_roi` 在拟合过程中发生了明显变化
  - 输出没有卡在平凡常数解
  - 当前 decoder-only 拟合路径至少在 `L=3` 上可以工作

Issue 5 是否可以认为最小完成：
- 可以。
- 就 Stage 3 / Issue 5 的目标来说，现在已经有了一个仓库内可复跑的 decoder-only single-sample fitting 入口，能够隔离 encoder、直接验证当前 optical decoder 栈的基本表达能力。
- 后续如果继续推进，下一步应进入更系统的 small-subset / depth comparison，而不是回头扩展本 issue 的范围。