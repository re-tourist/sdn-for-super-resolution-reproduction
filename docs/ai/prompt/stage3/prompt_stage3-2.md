你正在当前仓库中实现 Stage 3 / Issue 2。

任务背景：
这是一个 optical neural network 复现项目，目标论文是《Super-resolution image display using a diffractive optical network / decoder》。
当前不是重新设计系统，也不是移植 upstream 全项目，而是在现有 Stage 3 contract freeze 基础上继续实现。

你必须先严格遵循以下文档中的已冻结决策与边界：

- docs/plan/stage3_contract_freeze.md
- docs/integration/stage3_unresolved_params.md
- docs/integration/sdn_inventory.md
- docs/integration/sdn_gap_analysis.md
- docs/plan/stage3_sdn_porting_plan.md
- AGENTS.md

当前阶段定位：

- Stage 3 的正式目标是构建一个 paper-aligned optical decoder skeleton，并先验证 optical module 本身的可验证性
- 当前不是最终 end-to-end SR system
- 当前不是 final joint training
- 当前不是 final paper full-setting reproduction

本次只实现：
Issue 2 — optical propagation core

Issue 2 标题：
feat: extract paper-aligned propagation core from sdn_upstream

核心目标：
从 upstream 中抽取 propagation primitive，并在当前 repo 内重写为一个干净、独立、config-driven 的 optics core。
不要把 upstream 的任务层、数据层、head、electrical decoder 一起带进来。

你要完成的事情：

1. 从 upstream 参考并迁移 propagation kernel / FFT / IFFT / phase-mask modulation 的核心逻辑

2. 重写为当前 repo 内新的 optics 模块

3. 删除 classification head / electrical decoder / Pet 数据依赖 / OAM-LG-HG 任务耦合

4. 删除 upstream 中依赖 layer index 的隐式 final propagation 逻辑，尤其不能保留 `layer == 2` 之类硬编码

5. 使用显式的 layer semantics：

   - L 表示 trainable diffractive phase masks 数量
   - final propagation to sensor plane 不计入 L

6. 使用显式 distance schedule schema：

   ```yaml
   distance_schedule:
     input_to_first: ...
     inter_layer: [...]
     last_to_sensor: ...
   ```

1. 明确内部 field 表示，优先使用 complex tensor；
    如果不方便，也必须使用明确命名的 real/imag 表示，禁止继续沿用含混的 amp/phase 命名
2. 提供 clean 接口，为后续 Issue 3 的 readout / crop 做好衔接，但本次不要把 Issue 3 一起做完

当前已冻结的 contract，你必须遵守：

- 最小链路是：
   phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi -> loss/metrics
- 但本次 Issue 2 只负责把前半段做干净：
   phi_lr -> U0 -> U_out_full
- phase-only 主线默认振幅固定为 1
- full-grid 与 ROI supervision 的 contract 已冻结，但 crop/readout 是 Issue 3，不要在本次实现中偷渡完整损失链

建议落地目录（优先按此执行）：

- src/models/optics/phase_utils.py
- src/models/optics/propagation.py
- src/models/optics/diffractive_decoder.py

建议模块职责：

1. phase_utils.py
   - 提供 phase-only -> complex coherent field 的工具函数
   - 例如 `phase_to_field(phi)`，默认 amplitude=1
   - shape / dtype / device 处理要清晰
   - 不写任务层逻辑
2. propagation.py
   - 封装单次传播 primitive
   - 封装 phase mask modulation primitive
   - 不依赖数据集、训练器、electrical decoder、head
   - 不把 distance / grid 参数硬编码在源码里
3. diffractive_decoder.py
   - 构建 paper-aligned optical decoder skeleton
   - 支持 L=1/3/5
   - forward 主流程要清晰：
      a. phi_lr -> U0
      b. input_to_first propagation
      c. 逐层 phase mask modulation
      d. 层间 propagation
      e. last_to_sensor propagation
      f. 返回 U_out_full
   - 需要体现 config 驱动的 distance / grid / padding
   - 不包含 readout head / crop / loss / trainer

你必须避免的事情：

- 不要直接 vendoring 整个 upstream 目录到 src/
- 不要复制 upstream 的整套类结构和任务接口
- 不要引入 classification head
- 不要引入 electrical decoder
- 不要引入 reference readout
- 不要写 Pet / OAM / LG / HG 数据依赖
- 不要实现 end-to-end trainer
- 不要提前做 quantization / misalignment / hardware robustness
- 不要进行与当前任务无关的大规模重构
- 不要把配置写死在代码里

实现风格要求：

- 小步修改，尽量局部
- 配置优先，不写死数值
- 注释说明“哪些逻辑来自 upstream optics primitive，哪些是当前 repo 的 contract 重写”
- 对关键 shape、dtype、complex field 处理写清楚
- 保持 import 路径整洁
- 代码必须可读、可验证、可追踪

建议增加的最小自检：

- 一个随机 phase 输入能 forward 成功
- L=1/3/5 都能实例化
- 输出无 NaN / Inf
- 改变输入 phase 或 mask phase 后，U_out_full 会变化
- 不依赖任何 electrical decoder 才能运行

本次交付物：

- src/models/optics/phase_utils.py
- src/models/optics/propagation.py
- src/models/optics/diffractive_decoder.py
- 若有必要，补一个极小的 smoke test 或最小使用示例，但不要扩展成完整 script runner

验收标准：

- optical core 可独立 import 和 forward
- 不再出现 `layer == 2` 这类硬编码
- L 的语义正确
- distance_schedule 显式、清晰
- 返回结果是 clean 的 U_out_full
- 没有把 upstream 的任务层耦合带进来

输出要求：

1. 先简短说明你理解到的 contract 和本次实现边界
2. 再给出你准备修改/新增的文件列表
3. 然后实施代码修改
4. 最后给出简短的变更总结、风险点和建议下一步（应指向 Issue 3，而不是越级做 Stage 4）