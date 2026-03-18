实现了一个独立的 Stage 5 paper-aligned SR loss 路径：新增 `src/losses/`，把 `normalized MAE on I_out_roi + optional efficiency penalty` 从 Stage 4 脚本参考中抽离成可配置模块，并用小型 fake-batch sanity 脚本验证了 `sigma`、`eta`、`gamma_by_depth` 和 efficiency 开关行为。整个 issue 没有改 optics、encoder、trainer 或 eval。

精确变更文件：
- [src/losses/__init__.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/losses/__init__.py)
- [src/losses/stage5_sr_loss.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/losses/stage5_sr_loss.py)
- [configs/stage5/stage5_sr_loss.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_sr_loss.yaml)
- [scripts/check_stage5_sr_loss.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/check_stage5_sr_loss.py)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)

本次拍板的 Stage 5 `sigma` policy：
- 选择 `sigma_mode = per_sample`
- `sigma` 按每个样本的 ROI 非 batch 维求和计算：`sigma = sum(target_roi) / (sum(prediction_roi) + epsilon)`
- 先得到逐样本 `sigma`，再计算逐样本 normalized MAE，最后在 batch 维做 mean 聚合
- 这与 Stage 5 protocol freeze 的默认口径一致，并在 config 中显式记录为 `aggregation = batch_mean_after_per_sample_mae`

本次拍板的 Stage 5 `gamma_by_depth` policy：
- `L=1`: `gamma = 0.005`
- `L=3`: `gamma = 0.015`
- `L=5`: `gamma = 0.0`
- `L=5 = 0.0` 现在已经不再是隐含默认，而是显式写入 config 和 sanity summary 的 Stage 5 结论
- 该结论被记录为 Stage 5 default policy，不表述为论文唯一无歧义真值

`eta` 的实现口径：
- `eta = 100 * P_o / P_i`
- `P_o` 默认取 `prediction_roi` 在所有非 batch 维上的求和，也就是 `prediction_roi_sum`
- `P_i` 不从 hidden global 获取，而是通过 `input_power` 显式传入 loss
- 当 efficiency term 关闭时：
  - `total_loss = mae_term`
  - `efficiency_term = 0`
  - `eta = None`

sanity evidence：
- 运行命令：
  `python scripts/check_stage5_sr_loss.py --config configs/stage5/stage5_sr_loss.yaml`
- 生成 summary：
  [outputs/stage5/loss_sanity/stage5_sr_loss_summary.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/loss_sanity/stage5_sr_loss_summary.json)
- 记录内容包括：
  - `prediction_roi_shape = (2, 1, 96, 96)`
  - `target_roi_shape = (2, 1, 96, 96)`
  - `epsilon = 1e-8`
  - `sigma_mode = per_sample`
  - `gamma_by_depth = {1: 0.005, 3: 0.015, 5: 0.0}`
  - `eta_scale = 100`
  - `L=1/3/5` 下的 enabled/disabled loss、`mae_term`、`efficiency_term`、`sigma_mean`、`eta_mean`
- sanity 结果显示：
  - `L=1` 和 `L=3` 打开 efficiency term 后，总 loss 按预期增大
  - `L=5` 因 `gamma = 0.0`，打开/关闭 efficiency term 的总 loss 不变，这是刻意保留的 Stage 5 policy，不是遗漏

有意留给后续 issue 的未决项：
- trainer 里具体从哪一个上游张量派生 `input_power` 仍留给 `5.6` 做窄集成；本 issue 只冻结 loss 接口，不写 trainer
- smoke / main run 是否启用 efficiency term 作为运行策略，仍归 `5.9` / `5.10` 的 config 消费，不在这里扩成训练框架
- dataset、distance mapping、phase range、crop/FOV semantics 不在本 issue 内重开

已执行验证：
- `python -m py_compile src/losses/stage5_sr_loss.py scripts/check_stage5_sr_loss.py`
- `python scripts/check_stage5_sr_loss.py --config configs/stage5/stage5_sr_loss.yaml`

推荐的 commit 三段式信息：

```text
feat(train | stage5-5): add paper-aligned sr loss

why:
freeze the stage5 loss contract before trainer work so sigma, eta, and
gamma-by-depth are explicit, configurable, and no longer hidden inside
historical stage4 reference scripts

what:
add a stage5 super-resolution loss module with normalized mae on
I_out_roi, configurable efficiency penalty, explicit sigma and eta
definitions, stage5 gamma_by_depth defaults including L=5, and a fake-
batch sanity script with recorded toggle behavior
```
