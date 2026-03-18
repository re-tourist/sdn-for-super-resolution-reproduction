# Design Notes

记录当前仓库中已经实现并确认的 Stage 3 optical module 边界。重点是当前 stable contract，而不是未来愿景。

## 1. Stage 3 Stable Contract

当前 Stage 3 optical module 的稳定主链路为：

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi -> loss/metrics`

对应职责拆分：

- `phi_lr -> U0`
  - 由 `forward_from_phase(...)` 和 `build_input_field(...)` 负责
- `U0 -> U_out_full`
  - 由 `PropagationOperator` + phase-mask modulation 负责
- `U_out_full -> I_out_full`
  - 由 `intensity_readout(...)` 负责
- `I_out_full -> I_out_roi`
  - 由 `center_crop_2d(...)` 负责

当前 freeze 边界：

- full-grid forward + ROI supervision
- `U_out_full` 和 `I_out_full` 必须保留
- ROI crop 必须显式、确定性、可配置
- optical module 只负责 direct optical readout，不包含 electrical decoder

## 2. Propagation Primitive

当前实现位置：
- `src/models/optics/propagation.py`

来源与方式：
- 数学与数值思路来自 `external/sdn_upstream` 中值得复用的 optics primitive
- 当前 repo 内已重写为纯 torch、config-driven 实现

当前实际采用的 primitive：

- centered FFT / IFFT
  - `fft2_centered(...)`
  - `ifft2_centered(...)`
- Rayleigh-Sommerfeld transfer kernel
  - `build_rayleigh_sommerfeld_transfer_kernel(...)`
- 单次自由空间传播
  - `PropagationOperator`
- phase-only mask modulation
  - `apply_phase_modulation(...)`

当前明确去掉的内容：

- classification head
- electrical decoder
- Pet / OAM / LG / HG 任务耦合
- 基于 `layer == 2` 或 `depth + 1` 的隐式 final propagation 逻辑

## 3. L 与 Distance Schedule 语义

当前稳定定义：

- `L` 表示 trainable diffractive phase masks 数量
- final propagation to sensor plane 不计入 `L`
- 当前 Stage 3 支持 `L in {1, 3, 5}`

distance schedule 必须使用显式 schema：

```yaml
distance_schedule:
  input_to_first: ...
  inter_layer: [...]
  last_to_sensor: ...
```

对应语义：

- `input_to_first`
  - 输入面到第一层 trainable phase mask 的传播距离
- `inter_layer`
  - 相邻 trainable phase mask 之间的传播距离列表
- `last_to_sensor`
  - 最后一层 mask 到 sensor plane 的传播距离

当前 decoder 会显式检查 `inter_layer` 长度是否与 `L` 匹配，避免旧 upstream 的隐式 special case。

## 4. Grid / Padding / Readout / Crop

当前 grid contract 位置：
- `src/models/optics/diffractive_decoder.py`
- `src/models/optics/readout.py`

当前 grid 语义：

- `input_pattern_hw`
  - phase-domain 输入图样尺寸
- `layer_hw`
  - trainable diffractive phase mask 尺寸
- `propagation_hw`
  - optics propagation 使用的 full grid 尺寸

padding 规则：

- input pattern 和 phase mask 都会被 centered embed 到 `propagation_hw`
- center embed 是显式逻辑，不通过脚本临时补零

readout / crop 规则：

- `I_out_full = |U_out_full|^2`
- ROI 由 `ReadoutConfig.output_crop_hw` 控制
- ROI crop 采用 deterministic center crop
- crop 超出 full grid 时会显式报错

当前验证脚本使用的最小协议示例：

- `input_pattern_hw = (24, 24)`
- `layer_hw = (32, 32)`
- `propagation_hw = (48, 48)`
- `output_crop_hw = (20, 20)`

这些数值是 Stage 3 tiny validation setting，不应误读为 final paper-setting。

## 5. Phase Parameterization

当前 phase 相关实现位置：
- `src/models/optics/phase_utils.py`
- `src/models/optics/diffractive_decoder.py`

当前明确约定：

- 输入主线是 phase-only
- 默认 amplitude 固定为 1
- coherent field 使用 complex tensor 表示，不再沿用含混的 `amp/phase` 命名

当前两类 phase 处理责任不同：

- 输入 phase `phi_lr`
  - 由调用方直接提供 phase-domain tensor
  - decoder 在 `forward_from_phase(...)` 内部完成 `phase -> field`
  - 当前输入路径默认使用 identity phase mapping，然后构造 `exp(j * phi)`
- trainable phase masks
  - 内部保存 raw parameter
  - 通过 `map_phase(...)` 映射到物理 phase range
  - 当前默认配置是 `tanh` 映射到 `[-pi, pi]`
  - 当前默认初始化是 zero init

## 6. Entry Points and Responsibility Boundary

当前 decoder 的三个入口职责如下：

| 入口 | 输入语义 | decoder 负责什么 | caller / provider 负责什么 |
| --- | --- | --- | --- |
| `forward_from_phase(phi_lr, ...)` | real-valued phase tensor `[B, 1, H, W]` | `phase -> U0 -> propagation -> readout -> crop` | 提供 phase tensor，保证 batch/channel/spatial 语义正确 |
| `forward_from_field(U0, ...)` | complex coherent field `[B, 1, H, W]` | `propagation -> readout -> crop` | 提前构造好 complex field |
| `forward_from_phase_provider(provider, upstream_input, ...)` | provider 输出 phase | 调用 provider 获取 phase，然后委托给 `forward_from_phase(...)` | 提供满足 contract 的 provider |

额外说明：

- `forward(...)` 当前仍然直接委托到 `forward_from_phase(...)`
- 这样可以保持现有 Stage 3 Issue 3~6 脚本不需要改接口

## 7. Phase-Provider Hook

当前 hook 实现位置：
- `src/models/optics/phase_provider.py`

最小 contract：

```python
PhaseProvider(upstream_input: torch.Tensor) -> torch.Tensor
```

返回值要求：

- real-valued tensor
- shape 为 `[B, 1, H, W]`
- batch 维允许存在
- device / dtype 由 caller/provider 与 decoder 保持一致

当前 Stage 3 默认 provider：

- `DirectPhaseProvider`
  - 直接把调用方给的 phase tensor 原样返回

当前边界解释：

- 这是 future Stage 4 integration hook
- 它只定义“谁提供 phase”，不实现 encoder
- 它不包含 dataset、loss、trainer 或 joint training 逻辑

## 8. Stable Contract vs Future Hook

当前已稳定的 Stage 3 contract：

- `forward_from_phase(...)`
- `forward_from_field(...)`
- `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- `L` 的语义
- explicit distance schedule
- full-grid forward + ROI supervision
- intensity readout + center crop

当前只是为 future Stage 4 预留的 hook：

- `PhaseProvider`
- `DirectPhaseProvider`
- `forward_from_phase_provider(...)`

这些 hook 的作用仅仅是把 future encoder 接入点固定在 decoder 外部，而不是提前实现 Stage 4。

## 9. Known Issues and Current Limits

当前已知限制：

- Stage 3 主要完成了最小可运行与最小容量验证，不是 final paper reproduction
- 当前 tiny protocol 使用小 grid、短步数和 synthetic target，重点是 sanity / capacity，不是正式 benchmark
- 当前物理实现使用 float32 + centered FFT / RS kernel 的工程化版本，后续若进入更严格 paper alignment，仍需要继续核对参数与数值稳定性
- 当前没有引入 efficiency term、quantization、misalignment 或 hardware robustness
- 当前没有引入 encoder，也没有验证 Stage 4 闭环训练

因此，当前 design notes 的定位是：

- 固化 Stage 3 已实现并确认的接口边界
- 避免未来 Stage 4 误把 optical core 再次耦合回 encoder / trainer
- 为后续接入 learned encoder 提供最小而清晰的 phase-provider hook
