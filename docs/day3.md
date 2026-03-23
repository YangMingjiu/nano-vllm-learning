# Day 3 - KV Cache + PagedAttention

> 📅 日期：2025年3月13日
> 📍 阶段：阶段 2 - KV Cache + PagedAttention（Day 3）
> ⏱️ 学习时长：约 2h

---

## ✅ 今日完成

- [x] KV Cache 完整机制（每层缓存什么、Prefill 存入、Decode 追加）
- [x] RoPE 和 KV Cache 的交互（cache 存旋转后的 K，V 存原始值）
- [x] KV Cache 显存占用精确计算（通用公式 + LLaMA 系列对比）
- [x] KV Cache 三大内存问题（预分配浪费、外部碎片、无法共享）
- [x] PagedAttention 原理（Block、Block Table、逻辑/物理块映射）
- [x] PagedAttention 如何解决三大问题（按需分配、Block Pool、Copy-on-Write）
- [x] GQA 的含义及对 KV Cache 的影响
- [x] 现代模型（LLaMA-3、Qwen-2.5）与 LLaMA-2 的异同

---

## 💡 核心理解

### 1. KV Cache 完整机制

**缓存什么**：每一层的 K 和 V（不缓存 Q）。32 层 × 2（K和V）= 64 个 cache tensor。

**为什么不缓存 Q**：Decode 时只需要新 token 的 Q 去 attend to 所有历史的 K 和 V。旧 token 的 Q 不再使用。

**Prefill 阶段**：一次性算出所有 token 的 K 和 V，存入 cache。

```
K_cache: [1, 32, 6, 128]    ← 6 个 token 的 K
V_cache: [1, 32, 6, 128]    ← 6 个 token 的 V
```

**Decode 阶段**：每步只算 1 个新 token 的 q, k, v。k 和 v 追加到 cache，q 对全部 cache 做 Attention。

```
Decode step 1:
  K_cache: [1, 32, 6→7, 128]    ← 追加 1 个
  Attention: q [1,32,1,128] × K_cache^T [1,32,128,7] → [1,32,1,7]
  → 向量×矩阵 (GEMV)，不是矩阵×矩阵 (GEMM)
```

**RoPE 和 KV Cache 的交互**：cache 里存的 K 是已经经过 RoPE 旋转的。如果存旋转前的 K，每次 Decode 就得重新对 cache 中所有 K 施加 RoPE，失去缓存意义。V 不需要 RoPE，存原始值。

**核心 trade-off**：用显存换计算量。有 KV Cache 后，Decode 每步只需处理 1 个 token（而不是从头算全部 token）。

### 2. KV Cache 显存精确计算

**通用公式**：

```
单请求全部层 = 2 × n_layers × n_kv_heads × seq_len × d_head × bytes_per_element
```

**各模型对比（seq_len=4096, fp16）**：

| 模型 | n_layers | n_kv_heads | 单请求 KV Cache | 16 并发 |
|------|----------|------------|---------------|---------|
| LLaMA-2-7B (MHA) | 32 | 32 | 2.0 GB | 32 GB |
| LLaMA-2-13B (MHA) | 40 | 40 | 3.1 GB | 50 GB |
| LLaMA-2-70B (GQA) | 80 | 8 | 1.25 GB | 20 GB |

关键发现：70B 模型用 GQA 后，KV Cache 比 7B 还小（1.25 vs 2.0 GB），因为 KV 头数从 32 降到 8。

**seq_len 的影响**：KV Cache 和 seq_len 线性增长。seq_len=32768 时单请求就 16 GB，长上下文场景对 KV Cache 管理要求极高。

**和模型参数的对比**：LLaMA-2-7B 模型参数 14 GB 是固定开销，KV Cache 随请求数增长。16 个并发请求的 KV Cache（32 GB）比模型参数还大。

### 3. KV Cache 三大内存问题

| 问题 | 类比 | 原因 | 后果 |
|------|------|------|------|
| 预分配浪费（内部碎片） | 订大包间只来 2 人 | 不知道请求多长，按 max_seq_len 预分配 | 显存利用率低至 20-40% |
| 外部碎片 | 停车场到处小空位 | 不同请求长度/完成时间不同，释放后留下空洞 | 有空间但分不出连续块 |
| 无法共享 | 同一本书复印 4 份 | 连续内存分配无法让多个请求指向同一块 | Beam Search 等场景重复存储 |

**根源**：传统方式要求 KV Cache 在显存中连续存放。

### 4. PagedAttention 原理

**核心思想**：把操作系统虚拟内存的分页机制搬到 KV Cache 管理上。

```
OS 虚拟内存:                     PagedAttention:
  逻辑地址 → 物理地址               逻辑块号 → 物理块号
  页表做映射                        Block Table 做映射
  逻辑连续，物理不连续               KV 逻辑连续，物理散落各处
  按需分配页                        按需分配 Block
  Copy-on-Write                    Copy-on-Write（共享前缀）
```

**Block**：固定大小的 KV Cache 存储单元，存 block_size 个 token 的 KV。GPU 显存被预先划分成 Block Pool，空闲 Block 放在 Free List 中。

**Block Table**：每个请求有自己的 Block Table，记录逻辑块到物理块的映射。Attention kernel 根据 Block Table 去不同物理位置取 K 和 V。

**解决三大问题**：
- 预分配浪费 → 按需分配 Block，最多浪费最后一个 Block 的部分空间（<4%）
- 外部碎片 → Block 大小固定，任何空闲 Block 都能分给任何请求，碎片完全消除
- 无法共享 → 多个请求的 Block Table 可以指向同一个物理块（ref_count），修改时 Copy-on-Write

**效果**：KV Cache 利用率从 20-40% 提升到 >96%，吞吐量提升 2-4×（vs FasterTransformer）。

### 5. GQA（Grouped Query Attention）

多个 Q 头共享一组 KV 头：

```
MHA: 32 Q 头 : 32 KV 头    → KV Cache 最大
GQA: 32 Q 头 :  8 KV 头    → KV Cache 省 4 倍    ← 当前主流
MQA: 32 Q 头 :  1 KV 头    → KV Cache 最小，但质量损失明显
```

GQA 是 MHA 和 MQA 之间的最佳折中：显存大幅减少，质量几乎无损。现在几乎所有新模型（LLaMA-3、Qwen-2.5）都用 GQA。

### 6. 现代模型与 LLaMA-2 的异同

**不变的部分（所有现代 LLM 共享）**：Decoder-Only、自回归、Pre-Norm + RMSNorm、MHA/GQA 的 5 步流程、SwiGLU FFN、Residual、KV Cache、Prefill vs Decode。

**变化的部分（参数配置调优）**：
- GQA 成为标配（KV 头数越来越少）
- 词表变大（32K → 128K → 152K，支持更多语言）
- RoPE base 变大（10000 → 500000 → 1000000，支持更长序列）
- d_ffn 变大

核心架构 99% 一样，用 LLaMA-2 学到的推理流程直接适用于所有主流模型。

---

## 🆚 对比总结

### 传统 KV Cache 管理 vs PagedAttention

| | 传统方式 | PagedAttention |
|--|---------|---------------|
| 内存分配 | 预分配连续大块 | 按需分配固定大小 Block |
| 显存利用率 | 20-40% | >96% |
| 外部碎片 | 严重 | 完全消除 |
| 共享支持 | 不支持 | Copy-on-Write |
| Attention kernel | 标准（连续内存访问） | 修改版（按 Block Table 跳转访问） |

### MHA vs GQA vs MQA

| | Q 头数 | KV 头数 | KV Cache | 质量 |
|--|--------|---------|----------|------|
| MHA | 32 | 32 | 最大 | 最好 |
| GQA | 32 | 4-8 | 省 4-8× | 几乎无损 |
| MQA | 32 | 1 | 最小 | 有损失 |

---

## 🤔 今日反思

今天学完 KV Cache 和 PagedAttention 后，有一个感受：从阶段 2 开始，学的内容偏理论和概念，和阶段前做 GEMM、FlashAttention 项目时"写代码跑 benchmark"的体验很不一样。

意识到自己之前的项目并不是从零原创——GEMM 是跟着教程走的，FlashAttention 的 kernel 是 Claude 辅助生成的。但转念想，自己的价值不在于"我写了这段代码"，而在于"我理解了背后的原理"。能解释每个优化为什么有效、能算具体数字、能分析 trade-off——这才是面试考察的。

后续如果时间充裕，可以考虑做一个有"自己做决策"成分的小项目，弥补"独立解决问题"的经验缺口。候选方向：FA backward pass、简化版 Block Manager、或给 vLLM 提小 PR。

---

## 📚 额外收获

- 了解了国内 AI Infra 岗位分类：Kernel/算子优化、推理引擎、训练框架、AI 编译器、AI 芯片
- 自身定位：主攻 GPU Kernel 优化 + 推理系统方向，HPC + CUDA 背景 + 本科自动化的硬件知识是差异化优势
- 训练 vs 推理：训练是找参数（forward + backward + 更新），推理是用参数（只有 forward）。AI Infra 覆盖两者

---

## 🎯 Day 4 计划

1. vLLM 架构概览（Scheduler + Worker + BlockManager）
2. Continuous Batching（静态 vs 动态 batching）
3. 推理关键指标（TTFT / ITL / TPS / QPS / MFU）
4. SGLang 了解

→ 完成后阶段 2 收尾
