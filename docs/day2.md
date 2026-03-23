# Day 2 - Transformer 推理全流程（下）

> 📅 日期：2025年3月12日
> 📍 阶段：阶段 1 - Transformer 推理基础
> ⏱️ 学习时长：约 3h

---

## ✅ 今日完成

- [x] SwiGLU FFN（门控机制 + Swish 激活函数 + 数值例子）
- [x] Forward Pass 的概念（一次 forward = 走完全部 32 层）
- [x] 自回归生成（多次 forward pass，每次产生 1 个 token）
- [x] Prefill vs Decode 两阶段（Compute Bound vs Memory Bound）
- [x] KV Cache 的动机和显存压力分析
- [x] 输出投影 W_O 的意义（不同 head 信息融合）
- [x] 残差连接的原理（增量学习 + 梯度直通）
- [x] LM Head 的作用（hidden state → 词表概率分布）
- [x] 迷你模型完整端到端数值演算

---

## 💡 核心理解

### 1. SwiGLU FFN

```
SwiGLU(x) = (Swish(x × W_gate) ⊙ (x × W_up)) × W_down

三个权重矩阵:
  W_gate: [4096, 11008]   门控投影
  W_up:   [4096, 11008]   上投影
  W_down: [11008, 4096]   下投影
```

**Swish 激活函数**：Swish(x) = x × sigmoid(x)。核心行为是**大正值原样通过，其他值都被压到接近 0**。和 ReLU 相比过渡更平滑，小负值区域不是硬切到 0 而是缓缓趋近 0。

**门控机制**：gate 和 up 两条路径 element-wise 相乘。gate 值大 → "开"（保留特征），gate 值接近 0 → "关"（抑制特征）。和纯 sigmoid 做 gate 的区别：sigmoid 输出固定在 (0,1)，只能表达"开/关"；Swish 保留了正值的大小信息，既能"关"，又能表达"多重要"。类比：sigmoid 是 0/1 开关，Swish 是带音量旋钮的开关。

**d_ffn = 11008 而不是 16384**：SwiGLU 有 3 个矩阵（比传统 FFN 多 1 个），为保持总参数量相当，d_ffn 从 4×d_model 缩小到约 2.7×d_model。

**FFN 是每层计算量的大头**：FFN 占每层约 67% 的计算量，比 Attention 还多。但 FFN 全是 GEMM，适合 GPU 并行；Attention 有 Softmax 和 IO 瓶颈。

### 2. Forward Pass 和自回归生成

**一次 forward pass = 数据从输入到输出完整走过整个网络一遍**，包含 Embedding + 全部 32 层 Transformer Block + LM Head。不是 32 次 forward pass。

**自回归生成 = 多次 forward pass**，每次只产生 1 个 token。上一步的输出追加到下一步的输入：

```
Forward 1: "What is AI?"                    → "AI"
Forward 2: "What is AI?" + "AI"              → " is"
Forward 3: "What is AI?" + "AI is"           → " a"
...直到输出 [EOS]
```

模型不是"理解问题后一口气回答"，而是一个 token 一个 token 地续写，每一步都在做 next-token prediction。

### 3. Prefill vs Decode

| | Prefill | Decode（每步）|
|--|---------|-------------|
| 输入 | 完整 prompt（多个 token） | 1 个新 token |
| Q 的 shape | [n_heads, **seq_len**, d_head] | [n_heads, **1**, d_head] |
| 主要运算 | GEMM（矩阵×矩阵） | GEMV（向量×矩阵） |
| 瓶颈 | GPU 计算能力 → Compute Bound | HBM 带宽 → Memory Bound |
| 优化方向 | FlashAttention | KV Cache 管理（PagedAttention） |
| 对应指标 | TTFT | ITL / TPS |

**Decode 为什么是 Memory Bound**：每生成 1 个 token，要把整个模型参数（14GB）从 HBM 读一遍，但计算量很小（向量×矩阵）。99%+ 的时间在等数据。

**FlashAttention 主要优化 Prefill**：Prefill 时 S = QK^T 是大矩阵（seq_len × seq_len），IO 成本高，FA 通过 Tiling 避免写中间矩阵。Decode 时 S 只有 [1, seq_len] 一行，IO 本身不大，瓶颈在读 KV Cache。

### 4. KV Cache

**为什么需要缓存**：自回归生成中，Causal Mask 保证旧 token 的 K、V 每次结果完全一样，重复计算是浪费。

**缓存什么**：每一层的 K 和 V。不缓存 Q——Decode 时只需要新 token 的 Q 去 attend to 所有历史的 K、V。

**显存压力**：

```
LLaMA-2-7B, seq=4096, fp16:
  单请求 KV Cache = 2 × 32层 × 32头 × 4096 × 128 × 2B = 2 GB
  16 个并发请求 = 32 GB
  加上模型参数 14 GB → 46 GB → A100 40GB 装不下！
```

**KV Cache 三大问题**（→ PagedAttention 解决）：
1. 太大（并发一上来显存就爆）
2. 预分配浪费（不知道用户说多长，按 max_seq_len 预分配）
3. 内存碎片（不同请求长度/完成时间不同）

### 5. 输出投影 W_O 的意义

分头让每个 head 独立学不同的注意力模式。合头只是物理拼接，head 之间没有信息交互。W_O 的矩阵乘法让不同 head 的发现融合起来——分头是为了独立学习，W_O 是为了融合发现。

### 6. 残差连接

每层做两次残差：

```
X₁ = X + Attention(RMSNorm(X))     ← 保护 Attention
X₂ = X₁ + FFN(RMSNorm(X₁))         ← 保护 FFN
```

两个作用：
- **降低学习难度**：每层只需学"增量修正"，不需要学完整映射。如果某层学得不好（输出≈0），信息不丢失
- **梯度直通**：残差路径上没有非线性操作，梯度可以从第 32 层一路畅通流回第 1 层

### 7. LM Head

LM Head = 最后一个线性层，把 4096 维 hidden state 映射到 32000 维（词表大小），得到每个 token 的得分（logits）。经过 Softmax 后变成概率分布，argmax 或采样得到下一个 token。

有趣细节：很多模型的 LM Head 和 Embedding 层权重共享（weight tying），因为 Embedding 是 [32000, 4096]，LM Head 是 [4096, 32000]，恰好是转置关系。

---

## 🆚 对比总结

### Prefill vs Decode

```
Prefill: 计算 ████████████████████  IO ████████     → Compute Bound
Decode:  计算 ██                    IO ████████████████████  → Memory Bound
```

### 传统 FFN vs SwiGLU FFN

| | 传统 FFN | SwiGLU FFN |
|--|---------|-----------|
| 矩阵数 | 2 个 (W₁, W₂) | 3 个 (W_gate, W_up, W_down) |
| 激活函数 | ReLU | Swish |
| 门控 | 无 | 有（选择性激活特征） |
| d_ffn | 4 × d_model = 16384 | ≈ 2.7 × d_model = 11008 |
| 效果 | 好 | 更好 |

### sigmoid vs Swish 做门控

| | sigmoid | Swish |
|--|---------|-------|
| 输出范围 | (0, 1) | (-∞, +∞) |
| 大正值 | ≈ 1（固定上限） | ≈ x（保留大小信息） |
| 接近 0 | ≈ 0.5 | ≈ 0 |
| 大负值 | ≈ 0 | ≈ 0 |
| 表达能力 | 只能表达开/关 | 能表达开/关 + 强度 |

---

## 📊 完整端到端演算

今天生成了 `transformer_complete_walkthrough.md`，用迷你模型（d_model=4, n_heads=2, vocab_size=8, 1 层）走完了从 "I love AI" 到输出 token 的每一步数值计算：

```
Embedding → RMSNorm → QKV投影 → 分头 → RoPE → Attention(含Causal Mask + Softmax)
→ 合头 → W_O投影 → Residual → RMSNorm → SwiGLU FFN → Residual
→ Final RMSNorm → LM Head → Softmax → argmax → "AI"

然后演示了 Decode 阶段用 KV Cache 继续生成: "AI" → "is" → "great" → [EOS]
```

---

## 🤔 之前的误解纠正

| 误解 | 纠正 |
|------|------|
| 32 层 = 32 次 forward pass | 一次 forward pass 内部经过 32 层 |
| Swish 中小正值和大负值关，大正值和小负值开 | 只有大正值是"开"的，其他都接近"关" |

---

## 🎯 阶段 1 完成度

```
✅ Day 1: 架构全景 + RMSNorm + MHA 完整流程 + RoPE
✅ Day 2: SwiGLU FFN + Prefill vs Decode + 自回归生成 + KV Cache 动机

阶段 1 七个主题全部覆盖:
  [x] Decoder-Only 架构全景
  [x] RMSNorm
  [x] MHA 完整 5 步流程
  [x] RoPE 位置编码
  [x] SwiGLU FFN
  [x] Prefill vs Decode
  [x] 自回归生成 → KV Cache 动机

→ 阶段 1 基本完成！可以进入阶段 2: KV Cache + PagedAttention
```

---

## 💭 个人反思

今天把阶段 1 剩余的三个主题全部覆盖了，加上一份完整的端到端数值演算。

SwiGLU 的门控机制一开始不太好理解——最初以为 sigmoid 也能做门控（确实能），但搞清楚 Swish 相比 sigmoid 多保留了"正值的大小信息"这一点后就明白了。Swish 是带音量旋钮的开关，sigmoid 只是 0/1 开关，这个类比很好记。

Forward pass 的概念之前有混淆——以为 32 层就是 32 次 forward pass，实际上一次 forward pass 是走完全部 32 层。多次 forward pass 来自自回归生成的循环，每次产生 1 个 token。

Prefill vs Decode 的区别对我来说非常关键——它直接解释了为什么我做的 FlashAttention 主要优化 Prefill（大矩阵 IO），而 Decode 的瓶颈在 HBM 带宽（每步读全部参数，计算量却很小）。这个 Compute Bound vs Memory Bound 的区分也呼应了 GEMM 项目中的性能分析经验。

KV Cache 的显存计算让我直观感受到了问题的严重性——一个 7B 模型单请求就 2GB，16 个并发就 32GB。这让我非常期待下一阶段学习 PagedAttention 是如何用虚拟内存思想解决这个问题的。

阶段 1 到此基本完成，明天开始阶段 2！
