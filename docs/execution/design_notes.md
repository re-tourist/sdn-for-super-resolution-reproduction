# Design Notes

记录当前实现中的重要设计决策与限制。
这些内容不是 bug，但会影响未来模块开发。

---

## Evaluation Pipeline

### 1. Tensor shape ambiguity

当前 evaluation pipeline 支持以下输入：

- [H,W]
- [C,H,W]
- [N,C,H,W]

对于三维输入：


[N,H,W] / [C,H,W]


由于维度本身不可区分，当前实现采用启发式规则：


如果首维 ∈ {1,2,3,4}
→ 视为 channel 维


因此：


(2,28,28)


可能被解释为 2-channel image 而不是 batch=2。

建议未来调用 evaluation API 时统一使用：


[N,C,H,W]


以避免歧义。

---

### 2. Input range handling

evaluation pipeline 假设输入图像范围为：


[0,1]


当前实现的容错策略：

| 输入范围 | 行为 |
|---|---|
| [0,1] | 直接使用 |
| [0,255] | 自动除以 255 |
| 其它 | clip 到 [0,1] |

这种策略方便工具脚本，但可能掩盖上游错误。

未来正式实验建议：

- 明确保证输入范围为 `[0,1]`
- 或在 evaluation 中改为严格报错


### 1. Tensor shape ambiguity