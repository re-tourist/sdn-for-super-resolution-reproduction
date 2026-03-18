**边界**
本次只实现 Stage 4 最小 closed-loop trainer，用真实最小数据路径验证 `x_hr -> wrapper -> I_out_roi` 是否可学，并保存可检查工件。没有改 optical core，没有重写 dataset/adapter/wrapper，也没有进入 paper-final、Stage 5 或通用训练框架。

**文件**
本次新增文件：
- [scripts/train_stage4_minimal.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/scripts/train_stage4_minimal.py)

**变更总结**
脚本复用了现有 `Stage4EMNISTDataset`、`Stage4RoiTargetAdapter`、`MinimalHybridWrapper`，主链路固定为：
`x_hr -> wrapper -> I_out_roi`
`target_hr -> target_adapter -> target_roi`
`loss(I_out_roi, target_roi)`

实现内容包括：
- `--single-sample` 单样本 overfit 模式
- `--subset-size` 小子集 sanity 模式
- ROI-level normalized MAE 训练
- `freeze_encoder` / `freeze_optics` 显式冻结开关
- `config_snapshot.json`、`history.json`、`run_summary.json`
- `loss_curve.png`
- `preview_step0/best/final.png`
- `phi_preview_step0/best/final.png`
- `checkpoint_best.pt`、`checkpoint_latest.pt`
- `grad_stats.json`

**如何运行**
单样本 overfit：
```bash
python scripts/train_stage4_minimal.py --single-sample --steps 100
```

小子集 sanity：
```bash
python scripts/train_stage4_minimal.py --subset-size 4 --steps 100
```

可选冻结 optics：
```bash
python scripts/train_stage4_minimal.py --single-sample --steps 100 --freeze-optics
```

如果本地还没下 EMNIST，可补 `--download`。

**Artifacts**
默认输出目录：
- `outputs/stage4/minimal_trainer/run_<timestamp>/`

本地 smoke run 工件目录：
- [smoke_single](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/outputs/stage4/minimal_trainer/smoke_single)
- [smoke_subset](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/outputs/stage4/minimal_trainer/smoke_subset)

每次 run 至少会保存：
- `config_snapshot.json`
- `history.json`
- `run_summary.json`
- `loss_curve.png`
- `preview_step0.png`
- `preview_best.png`
- `preview_final.png`
- `phi_preview_step0.png`
- `phi_preview_best.png`
- `phi_preview_final.png`
- `checkpoints/checkpoint_best.pt`
- `checkpoints/checkpoint_latest.pt`
- `grad_stats.json`

**实际验证**
已完成本地 CPU smoke：
- `python -m py_compile scripts/train_stage4_minimal.py`
- single-sample 5 steps：`initial_loss=0.213564`，`final_loss=0.211453`，`encoder_grad_observed=True`，`optics_grad_observed=True`
- small-subset 3 steps：`initial_loss=0.233576`，`final_loss=0.232655`，`encoder_grad_observed=True`，`optics_grad_observed=True`

已确认：
- 短 run 可完整结束
- artifacts 成功落盘
- history / summary / grad stats 已记录
- 关键张量无 `NaN/Inf`
- single-sample 下 loss 有下降趋势

**资源建议**
本地 CPU 足够做 smoke 和工件检查。更长的 Stage 4 acceptance run 建议迁移到 Linux/GPU，例如：
```bash
python scripts/train_stage4_minimal.py --single-sample --steps 100 --device cuda
python scripts/train_stage4_minimal.py --subset-size 4 --steps 100 --device cuda
```

**下一步建议**
继续停留在 Stage 4 minimal acceptance：先用这个脚本完成正式的 single-sample overfit 和 small-subset 闭环验收，再把结果写入实验日志；不要直接跳到 Stage 5。