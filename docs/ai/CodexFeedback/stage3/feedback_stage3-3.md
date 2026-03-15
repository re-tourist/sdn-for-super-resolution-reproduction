Contract
本次严格按 frozen Stage 3 contract 只做 Issue 3 的 readout/crop 落地：把现有链路从 phi_lr -> U0 -> U_out_full 扩成 phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi。范围只到 direct optical readout，不回头重写 propagation core，也不提前做 trainer、joint training、single-sample fitting 或 capacity experiment。

Files

新增 [readout.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/readout.py)
修改 [diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py)
Implementation

在 [readout.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/readout.py):21 新增 ReadoutConfig，把 output_crop_hw 变成显式配置入口，并提供确定性的 center-crop slice 计算。
在 [readout.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/readout.py):52 实现 intensity_readout(field)，明确把 complex U_out_full 读成实值非负 I_out_full = re^2 + im^2。
在 [readout.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/readout.py):62 实现 center_crop_2d，crop 越界会显式报错，不再把切片逻辑散落到调用方或脚本里。
在 [diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py):165 给 decoder 增加 readout_config，但没有改 propagation.py 的数值核心。
在 [diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py):295 保留独立传播链 _propagate_from_field，把 U0 -> U_out_full 与 readout/crop 解耦。
在 [diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py):345 新增 readout_from_field，统一生成 I_out_full 和 I_out_roi。
在 [diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py):360 和 [diffractive_decoder.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/src/models/optics/diffractive_decoder.py):383 把 public forward_from_field / forward_from_phase / forward 接成显式返回结构，默认至少包含 U_out_full、I_out_full、I_out_roi；从 phase 入口还会保留 U0，return_intermediates=True 时继续附带传播中间量。
Closeout

变更总结：现在 optical module 已能完整输出 frozen Stage 3 前半链路，full-grid 和 ROI 被明确区分；crop 是显式 center crop，且由 output_crop_hw 控制。
最小自检已跑通：随机输入下 U_out_full 为 complex tensor，I_out_full/I_out_roi 为实值非负；output_crop_hw=(20,20) 和 (12,12) 时 ROI shape 正确变化；output_crop_hw=(64,64) 会显式报错；改变输入 phase 后，I_out_full 和 I_out_roi 都变化，平均差异约 1.75e-03 和 2.85e-03。
刻意后置的内容：normalized MAE / PSNR / SSIM 接口本身、Issue 4 的 forward sanity script、Issue 5 的 fitting loop、任何 electrical decoder / trainer / joint training。
说明：这次仍然没有把 smoke test 文件落到 tests/，因为此前 Windows sandbox 对新增 test 文件的 patch 继续不稳定；但代码已用内联命令完成同等验证。
下一步建议：直接进入 Issue 4，补正式的 L=1/3/5 forward sanity checks 和 full-grid / ROI 可视化检查，不要越级进入训练。

