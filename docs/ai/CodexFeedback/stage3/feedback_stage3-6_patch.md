我理解当前审核意见：Issue 6 之前的实现只能证明 `L=3` 下的小子集 decoder-only fitting 有效，但这还不等于完成了 Stage 3 freeze 里默认比较的 `L=1/3/5`。这次必须在同一 deterministic subset、同一 ROI-only normalized MAE、同一优化设置、同一步数预算下补齐 fixed-protocol depth sweep，才能把验收范围闭合。

我按这个边界做了最小补齐，没有改模型设计，也没有改 Stage 3 contract。具体改动是：
- 在 [train_decoder_only_small_subset.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/scripts/train_decoder_only_small_subset.py#L332) 抽出可复用的 `run_subset_validation(...)`，保留原有单-depth CLI，同时把 `subset_summary.json` 补齐为包含 `total_count`、`success_rate` 和 `protocol` 快照。
- 新增 [train_decoder_only_small_subset_sweep.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/scripts/train_decoder_only_small_subset_sweep.py#L1)，在固定 protocol 下顺序运行 `L=1/3/5`，为每个 depth 写独立目录，并统一生成 `sweep_summary.json` 和 `depth_comparison.csv`。
- 在 [README.md](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/README.md#L21) 增加了 sweep 入口说明。

本地已经直接跑完 sweep，输出目录是 [outputs/optics/decoder_only_small_subset_sweep_run1](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/outputs/optics/decoder_only_small_subset_sweep_run1)。统一汇总文件是 [sweep_summary.json](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/outputs/optics/decoder_only_small_subset_sweep_run1/sweep_summary.json) 和 [depth_comparison.csv](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/outputs/optics/decoder_only_small_subset_sweep_run1/depth_comparison.csv)。本次固定 protocol 为：`subset_size=4`、`steps=80`、`seed=42`、`device=cpu`、`lr_phase=0.15`、`lr_decoder=0.03`、`freeze_decoder=False`、`loss=normalized_mae_roi`。结果如下：
- `L=1`: `success_count=4/4`, `mean_initial_loss=0.125810`, `mean_best_loss=0.028643`, `mean_loss_decrease=0.097167`
- `L=3`: `success_count=4/4`, `mean_initial_loss=0.132044`, `mean_best_loss=0.032257`, `mean_loss_decrease=0.099787`
- `L=5`: `success_count=4/4`, `mean_initial_loss=0.134866`, `mean_best_loss=0.031355`, `mean_loss_decrease=0.103511`

这意味着 `L=1/3/5` 三个 depth 都在同一 protocol 下完成了多样本下降，且每个 depth 都有独立可追溯结果目录，不再是只靠 `L=3` 的局部结果支撑结论。就当前冻结验收项而言，Issue 6 现在可以重新申请关闭。另有工作区中已有的 `docs/ai/...` 非本任务改动我没有处理。