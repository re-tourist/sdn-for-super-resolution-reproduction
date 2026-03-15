你正在当前仓库中继续 Stage 3 implementation。

当前状态：
- Stage 3 planning freeze 已完成
- Issue 1（contract freeze）已完成
- Issue 2（optical propagation core）已完成
- 当前 optical core 已经能完成：
  phi_lr -> U0 -> U_out_full
- 现在进入 Issue 3 的实现，不要回头重写 Issue 2，也不要提前做 Issue 4/5

你必须严格遵循以下文档：
- AGENTS.md
- docs/plan/stage3_contract_freeze.md
- docs/integration/stage3_unresolved_params.md
- docs/plan/stage3_sdn_porting_plan.md
- docs/integration/sdn_gap_analysis.md

你必须接受以下 frozen contract，而不是重新解释：
1. Stage 3 最小链路固定为：
   phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi -> loss/metrics
2. Stage 3 固定采用 full-grid forward + ROI supervision
3. crop 必须是显式、确定性的 center crop
4. crop 逻辑不能隐藏在脚本临时逻辑中
5. output_crop_hw 等 crop 参数必须由配置控制
6. U_out_full 和 I_out_full 必须保留，不能只返回 ROI
7. 当前 Stage 3 optical module 只负责 direct optical readout，不引入 electrical decoder
8. 当前 Issue 3 只实现 readout/crop contract，不提前做 trainer、joint training、capacity experiment

本次任务：
Issue 3.3 — 在现有 optics core 上实现 output-plane intensity readout + ROI crop contract

本次目标：
把当前链路从
  phi_lr -> U0 -> U_out_full
扩展为
  phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi
并把 full-grid / ROI 的返回接口正式落地。

你要完成的核心工作：
1. 增加 intensity readout
   - 输入：complex field U_out_full
   - 输出：I_out_full = |U_out_full|^2
   - 必须显式、可读、无任务耦合
   - 不要把 intensity readout 混进训练脚本临时写法

2. 增加 ROI crop
   - 输入：I_out_full
   - 输出：I_out_roi
   - 默认采用显式 center crop
   - crop 参数必须来自配置，例如 output_crop_hw
   - 必须检查 crop 尺寸不能超过 full-grid 尺寸
   - 需要给出清晰、可复用的 crop spec / helper，而不是把切片散落在调用方

3. 区分 full-grid 和 ROI 输出
   - optical forward 结果中必须明确区分：
     - U_out_full
     - I_out_full
     - I_out_roi
   - 不允许只返回一个 intensity tensor 然后让调用方猜它是不是 crop 后结果

4. 为后续 normalized MAE / PSNR / SSIM 接口留好位置
   - 本次不要完整实现训练器
   - 但返回结构必须已经适合后续 loss-ready tensor 使用

推荐落地文件：
- src/models/optics/readout.py
- src/models/optics/diffractive_decoder.py
- 如确有必要，可小幅更新相关 config schema / dataclass
不要做无关目录重构。

推荐模块职责：
1. readout.py
   - intensity_readout(field) -> intensity
   - center_crop_2d(x, crop_hw) -> roi
   - 或者一个小型 ReadoutConfig / CropSpec helper
   - 职责要单一，不耦合 trainer / dataset / metrics

2. diffractive_decoder.py
   - 在现有 forward_from_phase / forward_from_field 基础上接上 readout
   - 保持现有 optical propagation core 不被污染
   - 返回结果建议为 dict 或清晰结构体，至少包含：
     {
       "U_out_full": ...,
       "I_out_full": ...,
       "I_out_roi": ...,
     }
   - 若现有 return_intermediates 已存在，应在不破坏现有语义的前提下扩展

必须遵守的实现边界：
- 不要重写 propagation.py 的核心数值逻辑，除非修一个明显阻塞 bug
- 不要引入 electrical decoder
- 不要引入 classification head
- 不要写 end-to-end trainer
- 不要提前实现 Issue 5 的 single-sample fitting loop
- 不要提前实现 quantization / misalignment / efficiency term
- 不要把 crop 逻辑藏到 scripts/ 里
- 不要把配置写死在源码里
- 不要因为方便而丢掉 U_out_full / I_out_full

建议加入的最小检查：
1. 对随机 phase 输入，能得到：
   - U_out_full: complex tensor
   - I_out_full: real nonnegative tensor
   - I_out_roi: real nonnegative tensor
2. I_out_roi 的 shape 与 output_crop_hw 一致
3. 当 crop_hw 改变时，ROI shape 正确变化
4. crop 居中逻辑正确，且不会 silently 越界
5. 若用户请求 full-grid 输出，不能只给 ROI
6. 改变输入 phase 后，I_out_full 和 I_out_roi 都会变化

建议配置约束：
- output_crop_hw 必须显式配置
- 如果已有 layer grid / propagation grid 配置，不要破坏现有 schema
- 若需要新增 readout 配置，保持最小化，不要发明复杂配置系统

代码风格要求：
- 小步修改，局部实现
- clear naming，禁止含混 amp/phase 命名继续污染接口
- 对 shape、dtype、complex/real 变换写清楚注释
- 所有新增逻辑应服务于当前 frozen contract
- 不要为未来 Stage 4 过度设计

本次交付物：
- src/models/optics/readout.py
- diffractive_decoder.py 中接入 readout/crop 的修改
- 如有必要，最小 config / dataclass 更新
- 若环境允许，补一个极小 smoke test 或最小示例；如果 tests 落地再次受 Windows sandbox 问题阻塞，则至少确保代码中保留可运行的最小自检入口思路，但不要编造已提交的文件

验收标准：
1. 当前 optical module 已能完整输出：
   phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi
2. I_out_full 与 I_out_roi 被明确区分
3. ROI crop 是显式 center crop，不隐藏在脚本中
4. crop 参数由配置控制
5. I_out_full / I_out_roi 可用于后续 loss-ready tensor
6. 没有引入 electrical decoder / classification head / joint trainer
7. 本次修改仍严格停留在 Stage 3 / Issue 3 范围内

输出要求：
1. 先简短复述你理解到的 frozen contract 与本次边界
2. 列出将新增/修改的文件
3. 再实施代码修改
4. 最后给出：
   - 变更总结
   - 尚未完成但刻意后置的内容
   - 下一步建议（应指向 Issue 4 forward sanity，而不是越级到训练）