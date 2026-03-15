已按你给的提示词 `prompt_stage1&2_step2.md` 严格完成 `Task1&2-3`，并只新增评估模块与测试脚本，没有改模型/训练/数据脚本。

1. 修改文件列表  
- 新增 [metrics.py](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\eval\metrics.py)  
- 新增 [evaluator.py](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\eval\evaluator.py)  
- 新增 [test_eval_pipeline.py](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\scripts\test_eval_pipeline.py)

2. 关键实现（按提示词要求）  
- `compute_psnr(pred, target)` / `compute_ssim(pred, target)`  
  - 支持输入：`[H,W] [C,H,W] [N,H,W] [N,C,H,W]`  
  - 支持类型：`numpy.ndarray` / `torch.Tensor`  
  - 内部统一：`float32 + [0,1]`  
  - SSIM 使用 `skimage.metrics.structural_similarity`，逐样本计算后取均值  
  代码：  
  - [metrics.py:25](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\eval\metrics.py:25)（维度统一到 NCHW）  
  - [metrics.py:40](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\eval\metrics.py:40)（range 归一化）  
  - [metrics.py:70](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\eval\metrics.py:70)（PSNR）  
  - [metrics.py:80](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\eval\metrics.py:80)（SSIM）

- 统一入口 `evaluate_batch(preds, targets)`  
  - 返回：`{"psnr_mean": float, "ssim_mean": float, "num_samples": int}`  
  代码：  
  - [evaluator.py:25](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\src\eval\evaluator.py:25)

- CLI sanity check 脚本  
  - 随机生成 HR，添加噪声得到 SR，调用 `evaluate_batch` 并打印指标  
  代码：  
  - [test_eval_pipeline.py:20](d:\AI project\ONN research\sdn-for-super-resolution-reproduction\scripts\test_eval_pipeline.py:20)

3. CLI 运行示例  
```bash
python scripts/test_eval_pipeline.py
```

我本地实测输出示例：
```text
PSNR: 22.30
SSIM: 0.9647
samples: 8
```

另外也验证了：
- `from src.eval.evaluator import evaluate_batch` 可正常导入
- numpy/torch 均可运行
- batch 输入可运行

4. 风险与说明  
- `range` 统一策略是：`[0,1]` 直接用；`[0,255]` 自动除以 255；其它范围做 `clip[0,1]`。  
  如果后续有特殊预处理（如 `[-1,1]`），建议在调用前先显式反归一化。  
- `[N,H,W]` 与 `[C,H,W]` 在 `N/C` 维度上存在天然歧义，当前用“首维是否像通道数(1/2/3/4)”做启发式判断。  
- 极小分辨率图像（边长 < 3）无法计算 SSIM，会抛出明确错误。