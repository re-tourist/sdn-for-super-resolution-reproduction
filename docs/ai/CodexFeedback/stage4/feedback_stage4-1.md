已补写 Stage 4 协议冻结文档，内容围绕最小闭环目标、Stage 3 继承边界、首版 tensor contract、训练验收协议和 Stage 5 准入证据展开，明确区分了“已冻结事实”“Stage 4 首版简化选择”“待确认项”。

变更文件：
- [docs/plan/stage4_protocol_freeze.md](d:/AI%20project/ONN%20research/sdn-for-super-resolution-reproduction/docs/plan/stage4_protocol_freeze.md)

关键冻结决策：
- Stage 4 只回答一个问题：接入最小 encoder 后，现有 optical decoder 能否在端到端梯度训练下学起来。
- Stage 4 最小闭环固定为 `HR input -> minimal encoder -> phi_lr -> optical decoder -> I_out_roi -> loss`。
- Stage 3 optical contract 直接继承，不允许改写 `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`。
- Stage 4 主训练入口固定走 `forward_from_phase(...)` 或 `forward_from_phase_provider(...)`，不以 `forward_from_field(...)` 作为主线。
- `phi_lr` 被冻结为 phase-domain tensor，不是通用 latent；raw-to-phase 映射责任在 encoder / wrapper，不在 optical decoder。
- 首版系统不允许在 optical decoder 后追加电子 refinement head。
- 首版主读出固定为 `I_out_roi`，主损失固定为 ROI `normalized MAE`；`PSNR/SSIM` 仅作辅助日志，不作验收门槛。
- 首版正式验收固定分两步：真实单样本 overfit、真实小子集闭环训练；Stage 3 的 synthetic / toy 脚本不计入 Stage 4 通过证据。
- Stage 4 trainer 必须保存 `summary.json`、`metrics.json`、`checkpoints/`、`samples/`，并记录 encoder 与 optics 的梯度可观测性。
- 进入 Stage 5 前，必须已有通过的真实单样本闭环和真实小子集闭环证据。

显式假设 / 待确认：
- `H_hr / H_phi / H_prop / H_roi` 的精确数值尚未冻结。
- 首版默认 smoke depth 选 `L=1` 还是 `L=3` 尚未冻结。
- 是否采用与当前电子 baseline 相邻的 `96 -> 24 -> 96` 空间尺度仍为 `待确认`。
- encoder 内部 raw output 到最终 `phi_lr` 的 phase range 映射尚未冻结。
- `target_hr -> target_roi` 的具体空间对齐实现尚未冻结。