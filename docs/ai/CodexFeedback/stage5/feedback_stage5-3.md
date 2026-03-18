实现了 Stage 5 的 paper-aligned optics 配置路径：新增了 `L=1/3/5` 的显式 optics config、一个 Stage-5 专用 forward-sanity 脚本，并生成了可审计的 summary evidence。整个 issue 只落在 optics configuration、distance mapping 和 crop/FOV alignment，没有改写 `DiffractiveDecoder` 或扩张到 encoder、loss、trainer、eval。

精确变更文件：
- [configs/stage5/stage5_optics_paper_aligned.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_optics_paper_aligned.yaml)
- [scripts/check_stage5_optics_configs.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/check_stage5_optics_configs.py)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)

最终拍板的 distance mapping：
- `L=1`：`input_to_first = d1 = 6.667λ`，`inter_layer = []`，`last_to_sensor = d3 = 173.333λ`
- `L=3`：`input_to_first = d1 = 4λ`，`inter_layer = [d2, d2] = [53.334λ, 53.334λ]`，`last_to_sensor = d3 = 53.334λ`
- `L=5`：`input_to_first = d1 = 2.667λ`，`inter_layer = [d2, d2, d2, d2] = [66.667λ, 66.667λ, 66.667λ, 66.667λ]`，`last_to_sensor = d3 = 80λ`
- 该映射以显式 `distance_schedule` 写入 config，不再依赖隐式 layer-index 分支。

crop / FOV alignment 定义位置：
- 配置入口在 [stage5_optics_paper_aligned.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_optics_paper_aligned.yaml) 的 `readout` 段：
  `output_crop_hw: [96, 96]`、`crop_policy: center_crop`、`center_offset_hw: [152, 152]`
- 运行时语义继续复用 [readout.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/readout.py) 的 `ReadoutConfig.center_crop_slices(...)` 和 `center_crop_2d(...)`
- 本次在 Stage 5 明确拍板为：从 `400x400` sensor-plane intensity 上做确定性的中心裁剪到 `96x96`，无额外 learned offset 或 margin

forward-sanity evidence：
- 运行命令：
  `python scripts/check_stage5_optics_configs.py --config configs/stage5/stage5_optics_paper_aligned.yaml`
- 生成 summary：
  [stage5_optics_forward_summary.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/optics_sanity/stage5_optics_forward_summary.json)
- 记录内容包括：
  - `input_pattern_hw = 32x32`（作为 inherited default 读入，仍保持可配置）
  - `layer_hw = 200x200`
  - `propagation_hw = 400x400`
  - `output_crop_hw = 96x96`
  - `center_offset_hw = 152,152`
  - `L=1/3/5` 的完整 distance schedule
  - `U_out_full / I_out_full / I_out_roi` 的输出 shape，三种 depth 均为：
    - `U_out_full = (1, 1, 400, 400)`
    - `I_out_full = (1, 1, 400, 400)`
    - `I_out_roi = (1, 1, 96, 96)`

有意留给后续 issue 的未决项：
- `phi_lr` / `input_pattern_hw` 的 `32x32` 这里只作为 Stage 5 inherited default 消费，不宣称为论文唯一终值；最终拍板仍归 `5.4`
- encoder 的 phase mapping range 仍归 `5.4`
- `L=5 gamma policy` 与 `sigma normalization granularity` 仍归 `5.5`
- dataset protocol 与 `96x96` tiling rule 不在本 issue 内改动，继续以 `5.2` 结果为准
- optics core 本身未重写；本 issue 只在已有 contract 上补显式配置

已执行验证：
- `python -m py_compile scripts/check_stage5_optics_configs.py`
- `python scripts/check_stage5_optics_configs.py --config configs/stage5/stage5_optics_paper_aligned.yaml`

推荐的 commit 三段式信息：

```text
feat(optics | stage5-3): add paper-aligned optics configs

why:
freeze the stage5 optics geometry and distance mapping before later
encoder, trainer, and eval issues, and make the 200/400 grid plus
96x96 readout path explicit and auditable

what:
add stage5 paper-aligned optics config for L=1/3/5, finalize the
d1/d2/d3 mapping into input_to_first/inter_layer/last_to_sensor,
define the 400x400 to 96x96 center-crop policy, and add a forward
sanity script with recorded shape and distance evidence
```
