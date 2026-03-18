我对本任务的理解是：只做 Stage 3 的接口整理，把 optical core 与未来 encoder 解耦，但不实现 encoder、不实现 joint model / joint trainer，也不改 Stage 3 已冻结的 optical/readout/crop 行为。目标是稳定 `forward_from_phase(...)`、`forward_from_field(...)` 的语义，并补一个最小的 phase-provider hook，让未来 Stage 4 只需要“提供 phase”就能接入。

本次修改的文件：
- 新增 [phase_provider.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/phase_provider.py#L1)
- 修改 [diffractive_decoder.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py#L161)

**变更总结**
- 在 [phase_provider.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/phase_provider.py#L26) 新增了 `PhaseProvider` protocol，约定“给一个上游 tensor，返回一个 real-valued `[B, 1, H, W]` phase tensor”。
- 在 [phase_provider.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/phase_provider.py#L40) 新增了最小 stub `DirectPhaseProvider`，它只是把调用方直接给的 phase 原样返回，用来表达“当前 Stage 3 默认仍是外部直接提供 phase”。
- 在 [diffractive_decoder.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py#L275) 把 `build_input_field(...)` 的 contract 写清楚了：输入是 real-valued phase tensor，decoder 负责 phase-to-field，默认 amplitude=1。
- 在 [diffractive_decoder.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py#L397) 明确了 `forward_from_field(...)` 只接收 complex coherent field，并直接进入 propagation/readout，不再承担任何 phase mapping 责任。
- 在 [diffractive_decoder.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py#L440) 明确了 `forward_from_phase(...)` 是稳定的 Stage 3 相位输入入口。
- 在 [diffractive_decoder.py](/d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py#L480) 新增了 `forward_from_phase_provider(...)`，它只做一件事：通过 provider 取 phase，再委托给 `forward_from_phase(...)`。这就是 future Stage 4 的接入点，但它本身不是 encoder 实现。

**Phase-provider Hook**
接口约束现在是：
- provider 形态：可调用对象，满足 `PhaseProvider(upstream_input) -> phi_lr`
- provider 输出：real-valued `torch.Tensor`
- shape 语义：`[B, 1, H, W]`
- batch 维：允许，且由 provider/caller 负责保持一致
- dtype/device：provider 输出什么 dtype/device，decoder 就在该设备上继续执行，但 phase 必须是 real-valued
- phase mapping 责任：provider 只“提供 phase-domain tensor”，不负责变成 field；decoder 仍通过 `forward_from_phase(...)` 内部完成 `phase -> U0`
- 当前 Stage 3 默认用法：直接外部提供 `phi_lr`，或者用 `DirectPhaseProvider` 包一层后走 `forward_from_phase_provider(...)`
- future Stage 4 接入位置：learned encoder 只需要产出满足上述 contract 的 phase tensor，然后喂给 `forward_from_phase(...)` 或 `forward_from_phase_provider(...)`

**刻意未实现的 Stage 4 内容**
- 没有实现任何 encoder module
- 没有实现 joint model / joint trainer
- 没有实现 end-to-end training loop
- 没有改 loss、metrics、ROI supervision contract
- 没有把 dataset、trainer、loss 逻辑塞进 provider hook
- 没有重构 optics core 的 propagation/readout 数值逻辑

**验证与风险**
- `py_compile` 已通过：`phase_provider.py`、`diffractive_decoder.py`
- inline smoke 已通过：`forward_from_phase(...)`、`forward_from_field(...)`、`forward_from_phase_provider(DirectPhaseProvider(), ...)` 都能运行，且 provider 路径与直接 phase 路径输出一致
- 现有脚本回归已通过：
  - `scripts/train_decoder_only_single_sample.py --depth 3 --steps 2`
  - `scripts/train_decoder_only_small_subset_sweep.py --depths 1 --subset-size 1 --steps 2`
- 当前风险点主要不在代码，而在未来 Stage 4 接入时必须守住 contract：encoder 输出必须仍是 real-valued `[B,1,H,W]` phase tensor，不能把 field、loss 或训练控制逻辑混进 optical core。建议下一步就在 Stage 4 开始时，把 encoder 输出直接接到这次新增的 phase-provider contract 上，而不是再改 decoder 内部接口。