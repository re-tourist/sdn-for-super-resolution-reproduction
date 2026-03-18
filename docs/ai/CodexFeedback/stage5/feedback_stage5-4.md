实现了一个真实的 Stage 5 paper-aligned phase-only encoder 路径：新增 `PaperPhaseEncoder`，用显式配置拍板 `phi_lr = 32x32` 和 `phase_range = [0, 2π)`，并用小型 sanity 脚本验证了 `96x96 -> phi_lr -> DiffractiveDecoder.forward_from_phase(...)` 的闭环集成。整个 issue 没有改写 optical core，也没有扩张到 loss、trainer 或 eval。

精确变更文件：
- [src/models/encoders/paper_phase_encoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/encoders/paper_phase_encoder.py)
- [src/models/encoders/__init__.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/encoders/__init__.py)
- [configs/stage5/stage5_paper_encoder.yaml](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/configs/stage5/stage5_paper_encoder.yaml)
- [scripts/check_stage5_encoder_integration.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/check_stage5_encoder_integration.py)
- [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)

本次拍板的 Stage 5 encoder 决策：
- 选择的 Stage 5 `phi_lr` size 为 `32x32`
- 选择的 Stage 5 `phase_range` 为 `[0, 2π)`，配置中写为 `[0.0, 6.283185307179586]`
- 这两个结论都被记录为 Stage 5 的 encoder/optics alignment default，而不是论文唯一真值
- 当前实现把 phase mapping 显式放在 encoder 侧，采用 `phase_mapping = sigmoid` 将 raw phase logits 映射到配置区间

实现细节边界：
- encoder 输出仍是 real-valued phase-domain `phi_lr`，没有输出 field-domain tensor
- `phi_lr -> U0` 仍由 [src/models/optics/diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py) 负责
- 没有改动 [src/models/hybrid/minimal_hybrid_wrapper.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/hybrid/minimal_hybrid_wrapper.py)，因为这次用直接的 `encoder -> decoder.forward_from_phase(...)` sanity 路径就足够证明集成成功

sanity / integration evidence：
- 运行命令：
  `python scripts/check_stage5_encoder_integration.py --config configs/stage5/stage5_paper_encoder.yaml`
- 生成 summary：
  [outputs/stage5/encoder_sanity/stage5_encoder_integration_summary.json](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/outputs/stage5/encoder_sanity/stage5_encoder_integration_summary.json)
- 记录内容包括：
  - encoder input shape：`(1, 1, 96, 96)`
  - raw phase shape：`(1, 1, 32, 32)`
  - `phi_lr` shape：`(1, 1, 32, 32)`
  - configured phase range：`[0.0, 6.283185307179586]`
  - observed `phi_lr` range：约 `(3.529985, 3.533639)`
  - downstream decoder shapes for `L=1/3/5`：
    - `U0 = (1, 1, 400, 400)`
    - `U_out_full = (1, 1, 400, 400)`
    - `I_out_full = (1, 1, 400, 400)`
    - `I_out_roi = (1, 1, 96, 96)`

有意留给后续 issue 的未决项：
- distance mapping 和 crop / FOV alignment 已由 `5.3` 拍板，本 issue 只消费现有 optics config，不重开
- loss semantics、`L=5 gamma`、`sigma normalization granularity` 仍归 `5.5`
- trainer architecture 和 eval protocol 仍归 `5.6` / `5.7` / `5.8`
- 本次没有把 `32x32` 或 `[0, 2π)` 叙述成论文唯一无歧义事实，只把它们固定为 Stage 5 的默认实现选择

已执行验证：
- `python -m py_compile src/models/encoders/paper_phase_encoder.py scripts/check_stage5_encoder_integration.py`
- `python scripts/check_stage5_encoder_integration.py --config configs/stage5/stage5_paper_encoder.yaml`

推荐的 commit 三段式信息：

```text
feat(model | stage5-4): add paper-aligned phase encoder

why:
freeze the stage5 encoder-side phi_lr contract before trainer and eval
work, and make the chosen 32x32 phase output plus explicit phase range
mapping auditable against the stage5 optics config

what:
add a paper-aligned phase-only encoder, expose target_hw and phase_range
in stage5 config, finalize the stage5 default phi_lr size and phase
range, and add an encoder-to-decoder sanity script with recorded range
and downstream shape evidence
```
