# Day 1 - Transformer 推理全流程（上）

> 📅 日期：2025年3月11日
> 📍 阶段：阶段 1 - Transformer 推理基础（Day 1 / ~7 days）
> ⏱️ 学习时长：约 3-4h

---

## ✅ 今日完成

- [x] Decoder-Only vs Encoder-Decoder 架构对比
- [x] Tokenizer + Embedding 流程（文本→token ids→向量）
- [x] Decoder-Only 完整 Pipeline 全景图（LLaMA-2-7B 参数）
- [x] RMSNorm 原理 + 数值例子 + 与 LayerNorm 对比
- [x] Pre-Norm vs Post-Norm
- [x] Multi-Head Attention 完整 5 步流程
- [x] RoPE 旋转位置编码原理 + 频率公式

---

## 💡 核心理解

### 1. 为什么是 Decoder-Only

所有 NLP 任务都可以转化为"给定上文，预测下文"——翻译、摘要、问答、代码生成全部统一为 next-token prediction。既然都是"续写"，就不需要 Encoder 和 Cross-Attention，一个 Decoder 搞定一切。架构更简单、更容易 scale up。

### 2. Token ≠ 单词

Tokenizer 切分的最小单元是 token，不是单词。高频词保持完整（如 "GPU" → 1 个 token），低频词拆成子词（如 "unhappiness" → "un" + "happiness" → 2 个 token）。LLaMA 还会自动在开头加 `<s>`（BOS token），所以 "The cat sat on the" 5 个单词变成 6 个 token。

- **seq_len**：tokenizer 切分结果决定的，不是人为定义的
- **batch_size**：我们控制的，同时处理几条序列
- **d_model = 4096**：模型架构设计时固定的

### 3. Embedding = 查表

Embedding Table 是一个 [32000, 4096] 的大矩阵。每个 token id 对应取一行。6 个 token 各取一行 → 得到 [1, 6, 4096] 的矩阵，这就是 X，整个网络的输入。表里的值是训练出来的，语义相近的词向量相似。

### 4. 32 层 = 相同结构、不同参数

每一层的结构完全一样（RMSNorm → Attention → Residual → RMSNorm → FFN → Residual），但每层有自己独立的权重参数。输入输出 shape 都是 [1, 6, 4096]——Residual Connection 要求输入输出维度相同，所以 4096 是贯穿全网络的"主干道宽度"。

### 5. RMSNorm vs LayerNorm

| 对比 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 减均值 | ✅ 有 | ❌ 去掉了 |
| 缩放参数 γ | ✅ | ✅ |
| 偏移参数 β | ✅ | ❌ 去掉了 |
| 计算量 | 多 | 少 |
| 效果 | 好 | 几乎一样好 |

RMSNorm 只保留"缩放"操作（除以 RMS × 可学习的 γ），去掉了 mean subtraction 和 bias。实验表明效果几乎无损，省计算。

Pre-Norm（先 Norm 再 Attention/FFN）让残差路径上没有非线性操作，梯度畅通，训练更稳定。

### 6. MHA 完整 5 步流程

```
X_norm [1,6,4096]
  │
  ├─ Step 1: QKV 投影 (3次GEMM)  → Q,K,V [1,6,4096]
  ├─ Step 2: 分头 (Reshape)       → Q,K,V [1,32,6,128]    零计算！
  ├─ Step 3: RoPE (只对Q和K)      → Q',K' [1,32,6,128]
  ├─ Step 4: Attention (FA!)      → O [1,32,6,128]         32头并行
  └─ Step 5: 合头+输出投影 (1次GEMM) → Attn_Out [1,6,4096]
```

关键理解：
- **W_Q, W_K, W_V 是权重（参数），Q, K, V 是计算结果**：Q = X_norm × W_Q
- **分头不是复制数据**：只是 reshape，把 4096 看成 32 组 × 128 维
- **每个 head 独立做 Attention**：可以学到不同的注意力模式（语法、语义、距离……）
- **W_O 的作用**：让不同 head 的信息混合交互
- **GEMM 计数**：Attention 部分共 4 次 GEMM（W_Q + W_K + W_V + W_O）

### 7. RoPE 旋转位置编码

**核心思想**：把位置编码为旋转角度。位置 m 的 token，每一对维度旋转 m × θ_i 弧度。两个 token 做点积时，旋转效果只取决于相对距离 (m-n)。

**频率公式**：θ_i = 10000^(-2i/d_head)
- 本质是从 1.0 到 ~0.0001 的指数衰减
- pair 0（θ≈1.0）：高频，像秒针，精确区分近距离
- pair 63（θ≈0.0001）：低频，像时针，区分远距离
- 不同频率的组合 → 同时感知近距离和远距离

**为什么只对 Q 和 K，不对 V？**
- Q·K^T 决定"谁该关注谁"（注意力权重）→ 需要位置信息
- V 是被加权求和的"内容" → 内容不应因位置而改变

**完整计算流程**：先对 Q 和 K 的每个 pair 做旋转 → 得到 Q' 和 K' → 然后 Q'·K'^T 点积中自然包含相对距离信息。

---

## 🆚 对比总结

### Encoder-Decoder vs Decoder-Only

| | Encoder-Decoder | Decoder-Only |
|--|---|---|
| 代表 | 原始 Transformer, T5 | GPT, LLaMA, Qwen |
| Attention 类型 | 双向 Self-Attn + Causal Self-Attn + Cross-Attn | 只有 Causal Self-Attn |
| 适用范式 | 翻译等 seq2seq | 统一的 next-token prediction |
| 当前趋势 | 逐渐被取代 | 主流 |

### 绝对位置编码 vs RoPE

| | 绝对位置编码 | RoPE |
|--|---|---|
| 注入方式 | Embedding 阶段加法 | 每层 Attention 前旋转 Q/K |
| 编码信息 | 绝对位置 | 相对位置（通过旋转角度差） |
| 额外参数 | 需要 | 零 |
| 外推能力 | 差（超过训练长度就失效） | 好（配合扩展技术） |

---

## 🤔 待深入 / 还没覆盖的

- [ ] SwiGLU FFN 的完整流程和门控机制
- [ ] Prefill vs Decode 两阶段的详细分析
- [ ] 自回归生成流程 → 引出 KV Cache 动机
- [ ] 只取最后一个位置预测 next token 的完整逻辑（已理解 Causal Mask 导致，但还没展开 Prefill/Decode 场景）
- [ ] RoPE 的外推问题和扩展技术（NTK-aware, YaRN 等，了解即可）

---

## 🎯 Day 2 计划

1. SwiGLU FFN 详解（门控机制 + 数值例子）
2. Prefill vs Decode 两阶段（Compute Bound vs Memory Bound）
3. 自回归生成 → KV Cache 的动机
4. 如果时间允许 → 开始整理阶段 1 总结笔记

---

## 💭 个人反思

今天是 Transformer 推理学习的第一天。之前只读过 The Illustrated Transformer，对原始 Encoder-Decoder 架构有一个模糊的印象，但对 Decoder-Only 的具体流程一直没有清晰的全景图。

今天最大的收获是**把 FlashAttention 项目放到了完整的上下文中**——原来我写的 FA kernel 处于 MHA 的 Step 4，输入是分头 + RoPE 之后的 Q'、K'、V，输出是每个 head 的 O。而 Step 1 的 QKV 投影和 Step 5 的输出投影就是 GEMM 项目直接对应的矩阵乘法。两个项目在架构中的位置现在非常清晰了。

一开始对 Token Embedding 的理解有偏差——以为 seq_len 是人为定义的，实际上是 tokenizer 切分决定的。搞清楚 "查表" 的含义后（就是根据 token id 取 Embedding Table 的对应行），整个从文本到向量的流程就通了。

RoPE 的频率公式 θ_i = 10000^(-2i/d_head) 一开始看着有点抽象，但用秒针/分针/时针的类比理解后，"不同频率覆盖不同距离范围"的直觉建立起来了。关键理解是：RoPE 不是直接"算出"相对距离这个数字，而是让相同相对距离的 token 对产生相同的旋转效果，模型通过训练学会利用这个信息。

明天继续 SwiGLU FFN 和 Prefill/Decode 两阶段，这部分会直接连接到 KV Cache 的动机，也就是通往 vLLM 的桥梁。
