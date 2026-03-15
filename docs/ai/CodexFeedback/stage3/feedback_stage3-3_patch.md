本次 closeout patch 修改了以下文件：

新增 [check_optical_readout.py](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/scripts/check_optical_readout.py)
更新 [README.md](d:/AI project/ONN research/sdn-for-super-resolution-reproduction/README.md)
新增的验证入口：

可复用脚本入口：python scripts/check_optical_readout.py
这个脚本会基于当前已经落地的 decoder API，重新执行 Issue 3 的 readout/crop 验证，不会改 propagation core，也不会引入 trainer、拟合流程或新的 Stage 3 contract。
这次任务没有再修改 diffractive_decoder.py，也没有重设计 decoder；只是把已有 Issue 3 行为固化成仓库内可重复运行的验证资产。
对 5 个必需检查项的覆盖情况：

U_out_full 是 complex tensor：已覆盖
I_out_full 和 I_out_roi 是实值且非负：已覆盖
修改 output_crop_hw 后，ROI shape 正确变化：已覆盖，脚本检查了 (20, 20) 和 (12, 12)
oversized crop 会显式报错：已覆盖，脚本断言会抛出 ValueError
修改输入 phase 后，I_out_full 和 I_out_roi 都会变化：已覆盖，脚本会断言差异非零并打印结果
当前状态：

新脚本已经实际运行通过，并完成了全部 5 项检查。
从 Issue 3 的收尾角度看，现在可以认为 Issue 3 已经可以干净关闭，因为 readout/crop contract 不再只依赖聊天记录或一次性 inline 命令，而是有了仓库内可复跑的验证入口。