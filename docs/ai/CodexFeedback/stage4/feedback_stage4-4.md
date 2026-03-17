本次边界已按要求收住：只实现 Stage 4 的最小模型壳子，把现有 minimal encoder 和现有 diffractive decoder 串起来；没有进入 dataset pipeline、ROI target adapter、trainer、loss loop，也没有改 Stage 3 optical core。

变更文件：
- [src/models/hybrid/minimal_hybrid_wrapper.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/hybrid/minimal_hybrid_wrapper.py)
- [src/models/hybrid/__init__.py](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/hybrid/__init__.py)

变更总结：
- 新增 `MinimalHybridWrapper`，内部只组合 `MinimalPhaseEncoder` 和 `DiffractiveDecoder`
- wrapper 主路径固定为 `x_hr -> phi_lr -> decoder.forward_from_phase(...)`
- forward 至少返回 `phi_lr`、`U0`、`U_out_full`、`I_out_full`、`I_out_roi`
- 可选返回 `raw_phase` 和 decoder 现有 intermediates，方便后续调试
- optical contract 保持不变，没有重写 `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`

wrapper 的 forward 输入输出说明：
- 输入：
  - `x_hr`: HR 域输入张量，语义为 `[B, 1, H_hr, W_hr]`
  - `amplitude`: 可选，原样透传给 decoder
  - `return_intermediates`: 是否透传 decoder 的中间传播张量
  - `return_encoder_debug`: 是否额外返回 encoder 的 `raw_phase`
- 输出：
  - 必有 `phi_lr`
  - 必有 `U0`
  - 必有 `U_out_full`
  - 必有 `I_out_full`
  - 必有 `I_out_roi`
  - `return_encoder_debug=True` 时额外返回 `raw_phase`
  - `return_intermediates=True` 时额外返回 decoder 既有调试张量

fake-batch 验证结果：
- 已执行 `python -m src.models.hybrid.minimal_hybrid_wrapper`
- fake HR batch 端到端 forward 成功
- 输出 shape 明确：
  - `phi_lr`: `(2, 1, 24, 24)`
  - `U0`: `(2, 1, 48, 48)`
  - `U_out_full`: `(2, 1, 48, 48)`
  - `I_out_full`: `(2, 1, 48, 48)`
  - `I_out_roi`: `(2, 1, 20, 20)`
- 自检中已检查关键张量无 `NaN/Inf`

下一步建议：
- 进入 Issue 4.3，补最小真实数据路径和 `target_roi` 适配
- 让 Issue 4.3 直接复用这个 wrapper 的 `I_out_roi` 输出作为后续监督接口
- 暂时不要直接跳到训练脚本；先把 `x_hr / target_roi` 的最小闭环数据边界补齐