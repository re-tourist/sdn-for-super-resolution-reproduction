已按最新 Stage 4 规划重做最小相位编码器，落点改为模块化目录 `src/models/encoders/`，并保留了显式 `phi_lr` 相位映射与轻量自检。联调 smoke 已通过：编码器输出可直接送入 `DiffractiveDecoder.forward_from_phase(...)`，返回了预期的 `U0 / U_out_full / I_out_full / I_out_roi`。

变更文件：
- [src/models/encoders/minimal_phase_encoder.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/encoders/minimal_phase_encoder.py)
- [src/models/encoders/__init__.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/encoders/__init__.py)

编码器输入/输出契约：
- Stage 4 主线输入：`x_hr`，单通道 HR 图像，shape 为 `[B, 1, H_hr, W_hr]`
- 模块构造参数暴露了 `in_channels`，默认值为 `1`
- 输出：`phi_lr`，单通道相位域张量，shape 为 `[B, 1, H_phi, W_phi]`
- 其中 `H_phi, W_phi` 由 `target_hw` 显式配置
- `phi_lr` 在进入 optical decoder 前已经具备 phase semantics，不是 generic latent

选择的 phase-range 映射：
- 使用显式有界映射
- `phi_lr = phase_center + phase_half_span * tanh(raw_phase)`
- 默认 `phase_range = (-pi, pi)`
- 具体实现留在 encoder 侧的 `map_raw_phase(...)`，没有放进 optical decoder

假设与延后事项：
- 这版是最小 Stage 4 encoder，不是论文最终 encoder
- `target_hw`、`base_channels`、`hidden_channels`、`phase_range` 仍保持可配置，未冻结为最终实验设定
- 仅实现 encoder 模块与最小自检，没有引入 trainer、dataset pipeline、hybrid wrapper 或 Stage 5 功能
- optical core 未改动

确认：
- 没有创建或保留 `src/models/stage4/` 目录
- 当前实现只使用了 `src/models/encoders/` 这一模块化路径