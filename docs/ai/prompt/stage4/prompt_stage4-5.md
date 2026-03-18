你正在当前仓库中实现：

Issue 4.5 — Implement Stage-4 minimal trainer with artifact saving

请先明确本任务边界：
- 这是 Stage 4 的最小 closed-loop trainer
- 当前目标不是 paper-final reproduction
- 当前目标不是大规模训练系统
- 当前目标不是 Stage 5 的 paper alignment
- 当前只做一个最小、可调试、可保存工件的训练脚本，用于验证：
  encoder + optical decoder 在真实最小数据路径下是否真的可学习

你必须严格遵循：
- docs/ai/PROJECT_CONTEXT.md
- docs/plan/plan_overview.md
- docs/plan/stage3_contract_freeze.md
- docs/paper/paper_notes.md
- 当前仓库中已落地的：
  - src/models/encoders/minimal_phase_encoder.py
  - src/models/hybrid/minimal_hybrid_wrapper.py
  - src/datasets/stage4_emnist.py
  - src/datasets/target_adapter.py
  - src/eval/metrics.py（若可复用）
  - scripts/train_electronic_baseline.py（训练骨架风格参考）
- 不重新设计 optical core
- 不改 Stage 3 frozen optical contract
- 不引入 quantization / misalignment / robustness / paper-final settings
- 不把当前 minimal protocol 伪装成论文最终实验

当前已知稳定前提：
1. Stage 3 optical contract 已冻结：
   phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi
2. MinimalPhaseEncoder 已能输出 phase-semantics 的 phi_lr
3. MinimalHybridWrapper 已能跑：
   x_hr -> encoder -> phi_lr -> decoder.forward_from_phase(...) -> I_out_roi
4. Stage4EMNISTDataset + Stage4RoiTargetAdapter 已完成：
   - batch["x_hr"]
   - batch["target_hr"]
   - target_adapter(target_hr) -> target_roi
5. 当前 Stage 4 的最小协议是：
   - x_hr 为真实 EMNIST 单通道 HR 图像
   - target_hr 当前与 x_hr 指向同一张真实 HR 图像
   - target_roi 由 target_hr 显式 resize/clamp 到 optical ROI
   - 这只是 minimal learnability protocol，不是 paper-final data protocol

本次任务目标：
实现第一个最小 Stage 4 trainer 脚本，支持：
1. single-sample overfit 模式
2. small-subset training 模式
3. artifact saving
4. history / summary / gradient statistics 记录
5. 训练后可人工检查 input / target / output / phase / loss curve

推荐新增文件：
- scripts/train_stage4_minimal.py

如有必要，可新增极少量辅助文件：
- 一个最小 config 文件
- 一个非常小的 artifact helper
但不要搭复杂 trainer framework。

你要完成的核心工作：

A. 搭建最小训练主循环
要求：
1. 直接复用现有模块：
   - dataset: Stage4EMNISTDataset / builder
   - target adapter: Stage4RoiTargetAdapter
   - model: MinimalHybridWrapper
2. 主链路必须清晰：
   x_hr -> wrapper -> I_out_roi
   target_hr -> target_adapter -> target_roi
   loss(I_out_roi, target_roi)
3. 优先采用当前项目已冻结且兼容 Stage 3 的 ROI-level normalized MAE
4. 不要新发明复杂训练框架
5. 训练脚本应尽量沿用 trusted electronic baseline 的风格，但不要强行抽象复用到过度复杂

B. 支持 single-sample overfit 模式
要求：
1. 提供显式 CLI 开关或模式参数
2. single-sample 优先级高于 small-subset
3. 至少要能在一张图上跑一个短训练并保存 before/after 结果
4. 若 single-sample 都不通，不要硬往 small-subset 扩

C. 支持 small-subset training 模式
要求：
1. 提供 very small subset（例如 4 或更小）的最小训练模式
2. 只作为 second-step sanity，不做大规模 sweep
3. 目标是验证 closed-loop 训练是否延续可学性，不追求最终指标

D. 保存 artifacts
至少保存以下内容到清晰的输出目录：
1. config_snapshot.json 或等价配置快照
2. history.json / csv，至少包含：
   - step / epoch
   - train loss
   - 若有 val，则记录 val loss
3. run_summary.json
4. loss_curve.png
5. preview 图：
   - x_hr
   - target_roi
   - pred I_out_roi
   - 可选：difference map
6. phase artifact：
   - phi_lr preview 或 tensor stats
   - 如方便，可保存 raw_phase preview
7. checkpoint：
   - 最小 model checkpoint（至少 latest 或 best 二选一）
8. gradient statistics：
   - encoder 参数梯度是否非零
   - optics 参数梯度是否非零（若当前设置允许 optics 训练）
   - 建议保存简单 summary，不必做复杂全层日志

E. 明确训练参数与冻结策略
要求：
1. 显式支持：
   - freeze_optics = true/false
   - freeze_encoder = false（默认 encoder 要训练）
2. 明确记录 optimizer 配置
3. 明确记录 single-sample / subset-size / steps
4. 若使用 toy optics grid 或 minimal HR/ROI 设置，必须在 summary 中写清是 minimal Stage 4 protocol

F. 最小验收检查
至少检查并汇报：
1. 一个短 run 能完整结束
2. artifacts 被成功保存
3. loss/history 被记录
4. 可视化文件可读
5. single-sample 模式下，loss 相比初始值有下降趋势
6. encoder 参数收到梯度
7. 输出与关键张量无 NaN/Inf

必须遵守的边界：
1. 不改 diffractive decoder 的物理实现
2. 不回头重写 dataset path、target adapter 或 wrapper，除非修复明确阻塞 bug
3. 不做 full paper hyperparameter alignment
4. 不做 quantization / misalignment / line-pair / robustness
5. 不引入 electrical decoder、额外 refinement head 或 paper-final trainer
6. 不把 Stage 4 minimal trainer 写成庞大的通用训练框架
7. 不编造训练结果；如果本地环境跑不完，要诚实说明，并给出迁移到 Linux 服务器的建议命令

实现风格要求：
- small, local, readable changes
- 优先可调试性、可追踪性
- clear naming
- 在注释和输出中明确：
  当前只是 Stage 4 minimal closed-loop learnability check
- 输出目录命名清晰，方便后续实验日志与文档引用

建议 CLI（可按仓库风格调整）：
1. single-sample:
   python scripts/train_stage4_minimal.py --single-sample --steps 100
2. small-subset:
   python scripts/train_stage4_minimal.py --subset-size 4 --steps 100
3. 可选冻结 optics:
   python scripts/train_stage4_minimal.py --single-sample --steps 100 --freeze-optics

建议输出目录结构（示例）：
outputs/stage4/minimal_trainer/
  run_<timestamp>/
    config_snapshot.json
    history.json
    run_summary.json
    loss_curve.png
    preview_step0.png
    preview_final.png
    phi_preview.png
    checkpoint_latest.pt
    checkpoint_best.pt
    grad_stats.json

输出要求：
1. 先简短复述你理解到的 Issue 4.5 边界
2. 列出准备新增/修改的文件
3. 再实施代码修改
4. 最后给出：
   - 变更总结
   - 如何运行 single-sample overfit
   - 如何运行 small-subset sanity
   - artifacts 保存位置与内容
   - 实际验证了什么
   - 若本地资源不足，哪些部分建议迁移到 Linux 服务器执行
   - 下一步建议（应仍停留在 Stage 4 minimal acceptance，而不是直接 Stage 5）