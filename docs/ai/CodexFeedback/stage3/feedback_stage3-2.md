本次我按 Stage 3 freeze 的边界实现了纯 optical core，范围严格停在 phi_lr -> U0 -> U_out_full。L 只表示可训练 diffractive phase masks 数量，最终到 sensor 的传播单独建模；没有带入 upstream 的 classification head、electrical decoder、Pet/OAM/LG/HG 耦合，也没有实现 Issue 3 的 readout/crop。

本次实际新增文件：

[phase_utils.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/phase_utils.py):154
[propagation.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/propagation.py):44
[diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py):158
变更内容：

[phase_utils.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/phase_utils.py):154 提供了显式 phase -> complex field 转换、可配置 phase mapping，以及 centered embedding。输入 field 和 mask 的语义分开处理，避免继续沿用 upstream 含混的 amp/phase 命名。
[propagation.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/propagation.py):44 把 upstream 的 centered FFT/IFFT、getH2 风格 Rayleigh-Sommerfeld kernel、phase-mask modulation 抽成独立 primitive，并重写成纯 torch、config-driven 的 PropagationOperator。
[diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py):158 新建了 paper-aligned decoder skeleton。这里用 DistanceSchedule、OpticalGridConfig、PhaseMaskConfig 明确了 distance_schedule.input_to_first / inter_layer / last_to_sensor、grid/padding 关系和 mask 参数化；forward_from_phase 与 forward_from_field 都只返回 U_out_full，但保留了 return_intermediates 方便后续接 Issue 3。
验证结果：

我用内联 smoke check 跑通了 L=1/3/5 的独立 import 和 forward，输出是 complex tensor，且无 NaN/Inf。
改变输入 phase 后，U_out_full 发生变化，平均差异约 2.27e-03。
改变第一层 mask raw phase 后，U_out_full 发生变化，平均差异约 3.85e-03。
没有提交仓库内的 smoke test 文件；apply_patch 在新增 tests/test_optics_core_smoke.py 时反复触发 Windows sandbox refresh 错误，所以这部分只做了命令级验证。
风险点：

当前只完成了前半链路，I_out_full / I_out_roi 还没接上，Issue 3 才能真正把 Stage 3 frozen readout/crop contract 落完整。
传播数值现在是按 upstream primitive 重写后的 clean version，仍需要在 Issue 4 的 forward sanity 中继续做更系统的 shape / crop / visualization 检查。
下一步建议直接进入 Issue 3：在这个 core 上补 U_out_full -> I_out_full -> I_out_roi 的显式 readout/crop contract，不要提前跳到 Stage 4 的 joint training。


