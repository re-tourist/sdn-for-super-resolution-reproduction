本次 Issue 4 我只做了最小、可复跑的 forward-sanity 资产，没有改 decoder 接口，也没有碰 propagation core。

1. 修改的文件
- 新增 [check_optical_forward_depths.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/check_optical_forward_depths.py)
- 更新 [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md):18
- 这次没有修改 `src/models/optics/` 里的实现代码，只复用了当前已落地的 Stage 3 decoder contract。

2. 新增的 sanity 入口
- 入口脚本：`python scripts/check_optical_forward_depths.py`
- 作用：对 `L=1/3/5` 统一执行前向 sanity 验证，检查输出 contract、dtype、非负性、ROI shape、一致性、跨深度差异，以及显式 distance schedule 语义。
- README 已补一行运行入口，方便后续复跑。

3. `L=1/3/5` 是否都运行成功
- 是，三种深度都已实际运行通过。
- 脚本输出确认：
  - `L=1`、`L=3`、`L=5` 都成功 forward
  - `U_out_full / I_out_full / I_out_roi` 全部存在
  - `I_out_full` shape 都是 `(1, 1, 48, 48)`
  - `I_out_roi` shape 都是 `(1, 1, 20, 20)`

4. 所有 required sanity checks 是否通过
- 是，全部通过。
- 已覆盖并实际通过的检查包括：
  - forward pass 可运行
  - 输出包含 `U_out_full`、`I_out_full`、`I_out_roi`
  - `U_out_full` 为 complex
  - `I_out_full`、`I_out_roi` 为 real 且非负
  - ROI shape 与 crop 配置一致
  - 三个深度的输出接口一致
  - 不同深度输出不平凡相同
  - 显式 distance schedule / layer semantics 生效：脚本额外验证了错误的 `inter_layer` 长度会直接报错，而不是依赖旧的隐式 layer-index 特判

5. Issue 4 是否可以认为已经 cleanly closed
- 可以。
- 就 Stage 3 / Issue 4 的目标来说，现在已经有仓库内可复跑的 sanity 入口，能稳定验证 `L=1/3/5` 的 forward contract 和 depth-aware 行为，不再依赖聊天记录或一次性 inline 命令。
- 下一步应进入后续计划里的训练前/调试前工作，不需要回头再补 Issue 4 的基础验证。