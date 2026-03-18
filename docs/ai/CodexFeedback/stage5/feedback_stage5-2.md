实现了一个可复现的 Stage 5 EMNIST `96x96` display 数据路径：新增了确定性合成数据集、一个小型 preview 脚本、Stage 5 数据配置示例，并实际生成了 train/val/test 的审计工件。这个 issue 只落在数据协议范围，没有触碰 optics、encoder、loss、trainer 或 eval 逻辑。

精确变更文件：
- [src/datasets/stage5_emnist_display.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/datasets/stage5_emnist_display.py)
- [src/datasets/__init__.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/datasets/__init__.py)
- [scripts/preview_stage5_emnist_display.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/preview_stage5_emnist_display.py)
- [docs/plan/stage_plan/stage5/stage5_emnist_display_config.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/docs/plan/stage_plan/stage5/stage5_emnist_display_config.yaml)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)

最终拍板的 `96x96` tiling rule：
- `96x96` display 固定为 `3x3` 个非重叠 `32x32` cells。
- 每个 occupied cell 恰好放 1 个由 EMNIST `28x28 -> 32x32` bicubic resize 得到的 letter。
- 同一张合成图里 occupied cells 从 9 个 cell 中无放回采样。
- letter 之间不重叠，empty cells 保持为 0。
- train/val 的 letter count 从 `{1,2,3,4}` 采样；test 从 `{6,7,8,9}` 采样。
- 当前实现显式修正了 EMNIST 常见的朝向问题，预览中是 upright letters。

split 生成和随机性控制：
- train/val 组合样本使用 EMNIST `letters` 的 train partition；test 使用 EMNIST `letters` 的 test partition。
- 样本量固定为 train `60000` / val `6000` / test `6000`。
- 每个 composite sample 都由稳定 seed 函数基于 `(base_seed, split, dataset_index)` 单独生成，所以访问顺序无关、同 seed 下可重现。
- preview 采样使用单独的 deterministic preview seed offset，不会改动数据集本身的生成规则。
- augmentation 只暴露 Stage 5 范围内的 rotation / flip / contrast；train 默认开启，val/test 默认关闭。

生成的 preview artifacts：
- [train preview](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/dataset_preview/train/preview_grid.png)
- [train summary](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/dataset_preview/train/summary.json)
- [val preview](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/dataset_preview/val/preview_grid.png)
- [val summary](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/dataset_preview/val/summary.json)
- [test preview](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/dataset_preview/test/preview_grid.png)
- [test summary](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/dataset_preview/test/summary.json)

有意留给后续 issue 的未决项：
- optics geometry 和 distance mapping，仍归 `5.3`
- phase mapping range，仍归 `5.4`
- gamma policy 和 sigma normalization，仍归 `5.5`
- output crop / FOV alignment，仍归 `5.3` / `5.7`

验证已执行：
- `python -m py_compile src/datasets/stage5_emnist_display.py scripts/preview_stage5_emnist_display.py`
- `python scripts/preview_stage5_emnist_display.py --config docs/plan/stage_plan/stage5/stage5_emnist_display_config.yaml --splits train val test --preview-count 4`

补充说明：
- 这个环境当前拒绝向 `configs/` 子树写入新文件，所以配置示例放在了 Stage 5 文档目录下，并由 preview 脚本直接读取。