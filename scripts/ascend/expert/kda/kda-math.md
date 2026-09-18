# KDA 语义与 chunked 等价形式

## 1. 逐 token 定义（评测参考实现）

```python
query = L2norm(q.float()); key = L2norm(k.float())        # x / sqrt(sum(x^2) + 1e-6)
query = query * scale                                     # scale = 1/sqrt(128)
decay = exp(lower_bound * sigmoid(exp(a_log)[:, None] * (g + dt_bias)))   # lower_bound = -5
gain  = sigmoid(beta)                                     # beta 是 raw logits
state = initial_state                                     # [B, H, V, K] fp32, (value, key)
for t in range(T):
    state *= decay[:, t, :, None, :]                      # 逐 value x 逐 key
    delta  = gain[:, t, :, None] * (v[:, t] - (state * key[:, t, :, None, :]).sum(-1))
    state += delta[:, :, :, None] * key[:, t, :, None, :]
    out[:, t] = (state * query[:, t, :, None, :]).sum(-1)
```

关键点：

- 衰减是 `[H, D]` 的**逐 key 维**因子，作用在 state 的 key 轴上（`state[v, k] *= decay[k]`）。
- `lower_bound=-5` 时 `decay = exp(-5 * sigmoid(...)) ∈ [e^-5, 1)`，永不爆炸。
- `final_state` 是 fp32、`[B, H, V, K]`（key 连续）；`output` 是 bf16 `[B, T, H, V]`。
- 归一化的 eps 在 sqrt 内部：`x / sqrt(sum(x^2) + eps)`，不是 `x / (sqrt(sum)+eps)`。
- 权重不随 token 变化、无 bias、无 mask；因果性来自 delta rule 本身。

## 2. chunked 形式（FLA `chunk_kda_fwd`，并行的来源）

每个 chunk 大小 BT（推荐 64；T=4096 可整除，无尾块）。令 `gk = cumsum(gate)/ln2`，
在 log2 域做指数（`exp2`）比自然指数快且与参考等价：

```text
Prepare（chunk 内，可完全并行）
  Aqk[i,j] = scale * <q_i * 2^gk_i, k_j * 2^-gk_j>       i >= j，否则 0
  Akk[i,j] = beta_j * <k_i * 2^gk_i, k_j * 2^-gk_j>      i > j （严格下三角）
  A        = (I + Akk)^-1   （下三角单位矩阵求逆）
  u        = A @ (beta * v)
  w        = A @ (beta * k * 2^gk)
  qg       = q * 2^gk
  kg       = k * 2^(gk_last - gk)        # gk_last = 本 chunk 末行

FwdH（跨 chunk 串行，head 内）
  v_new = u - w @ h
  o     = scale * (qg @ h) + Aqk @ v_new
  h     = h * 2^gk_last  +  kg^T @ v_new          # [K, V] fp32
```

- `(I + Akk)^-1` 用 Neumann/倍增：`A = (I - L)(I + L^2)(I + L^4)(I + L^8) ...`，
  下三角，BT=64 时乘到 L^32（5 次平方）。FLA 在 fp16 里做这些 dot（fp32 累加），
  精度敏感时可以退到 bf16/fp32。
- 只有 FwdH 串行；Prepare 对 chunk 完全并行，FwdH 对 `(head, V 分块)` 并行。
- 与逐 token 形式严格等价（浮点误差级别的差异），可用小 shape（BT=16、T=64）
  逐 token golden 对照验证。

## 3. 数值与 dtypes

| 量 | 精度 | 说明 |
|---|---|---|
| state `h` | fp32 | 必须；bf16 会在 4096 步上漂移 |
| gk（gate cumsum） | fp32 | cumsum 在 fp32 做，取出后转 bf16 参与 dot |
| Aqk/Akk/A | fp32 累加 | FLA 用 fp16 存 A；若精度不过关改 fp32 |
| 输出 o | bf16 | 与参考一致 |
| dot 输入 | bf16 | Cube 原生路径 |

- `2^gk` 与 `2^-gk` 都直接算，不要 `1/2^gk`（除零与精度都更差）。
- `exp2` 前的一切保持 fp32；只在 `tl.dot` 前转 bf16。
- 参考实现里 softplus 分支用不上：本算子固定 `lower_bound` 路径。

## 4. 参考实现的隐含契约

- 输入 `g`、`beta` 是 bf16/fp32 的 raw 值：gate 激活、beta sigmoid 都要在算子内部完成。
- `initial_state` 可能全零（fresh prefill），`final_state` 必须写回。
- `a_log[HEADS]`、`dt_bias[HEADS, HEAD_DIM]` 是 `[H]`/`[H, D]` 的 fp32 常量表，按 head 读取。
