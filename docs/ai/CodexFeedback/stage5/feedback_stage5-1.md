新增了 Stage 5 的协议冻结文档，把范围边界、继承 contract、paper-aligned 默认值、未决参数 ledger，以及后续 issue 的拍板权限都固定下来，便于后续实现 prompt 直接引用。

精确变更文件：
- [stage5_protocol_freeze.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/docs/plan/stage_plan/stage5/stage5_protocol_freeze.md)

新文档之外的 consistency edit：
- 无

最重要的冻结决策：
- Stage 5 明确继承并冻结 optical contract：`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
- Stage 4 learnability gate 已通过，Stage 4 artifacts 只作为 regression baseline，不在 Stage 5 重开
- Stage 5 是第一次对齐论文完整主设定的阶段；Stage 6 仍负责 quantization、misalignment/robustness、系统化 `L=1/3/5` ablation、complex/amplitude-only 对比
- Stage 5 当前默认目标已写清：EMNIST `96×96` protocol、`200/400` optics grid、phase-only encoder、paper loss、Adam/LR/batch/epoch targets、PSNR/SSIM + bicubic + blind line-pair eval path
- 每个未决项都绑定了唯一 owner issue，并明确哪些后续 issue 不得静默拍板

最重要的刻意保留未决项：
- `phi_lr` size：当前默认 `32×32`，由 `5.4` 最终拍板，之前必须保持可配置
- `96×96` tiling rule：由 `5.2` 最终拍板
- distance mapping 与 output crop/FOV alignment：由 `5.3` 最终拍板
- `L=5` gamma policy 与 sigma granularity：由 `5.5` 最终拍板
- phase mapping range：当前默认 `[0, 2π)`，由 `5.4` 最终拍板

这是文档任务，未运行测试。