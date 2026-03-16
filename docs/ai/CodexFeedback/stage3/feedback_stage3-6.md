本次严格停留在 Stage 3 / Issue 6 的边界内，只补了一个可复跑的小子集 decoder-only 容量验证入口，没有回头改 Issue 5，也没有引入 encoder、joint training 或 decoder 重设计。

修改的文件：
- 新增 [train_decoder_only_small_subset.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/scripts/train_decoder_only_small_subset.py#L1)
- 更新 [README.md](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/README.md#L15)

新增的可运行入口：
- `python scripts/train_decoder_only_small_subset.py --depth 3 --subset-size 4 --steps 80`
- 脚本会对一个确定性小子集逐样本执行 decoder-only fitting：每个样本单独创建可学习输入 phase，走当前 `I_out_roi` 链路，用 ROI 上的 normalized MAE 优化，并保存每个样本的 `summary.json`、`artifacts.pt`、`roi_triptych.png`、`loss_curve.png`，同时输出总体 `subset_summary.json`。

本次实际验证配置：
- 子集大小：`4`
- depth：`L=3`
- steps：`80`
- loss：Stage 3 当前默认的 ROI-only `normalized MAE`

验证结果：
- `L=3`、`subset_size=4` 的脚本已实际跑通。
- 4/4 个样本都出现了明确的 loss 下降，不是单个幸运样本。
- 汇总结果：
  - `mean_initial_loss = 0.132044`
  - `mean_best_loss = 0.032257`
  - `mean_loss_decrease = 0.099787`
- 每个样本的 `roi_change_mean_abs` 都明显大于 0，说明 `I_out_roi` 在优化过程中发生了实质变化，而不是卡在平凡常数解。
- 全流程保持 encoder-free，只优化输入 phase，并沿用现有 optical decoder 路径。

结论：
- 就 Stage 3 / Issue 6 的目标而言，现在已经有了仓库内可复跑的小子集容量验证入口，能够证明当前 decoder-only 路径可在多个目标上重复使用并稳定下降。
- 因此，Issue 6 可以认为已经最小完结。