# Day 4 - 阶段 2 收尾 + 阶段 3 广度补充 + 项目规划

> 📅 日期：2025年3月16日
> 📍 阶段：阶段 2 收尾 + 阶段 3 全部完成
> ⏱️ 学习时长：约 2h

---

## ✅ 今日完成

### 阶段 2 收尾
- [x] vLLM 架构概览（Scheduler + BlockManager + Worker）
- [x] Continuous Batching（静态 vs 动态、iteration-level 调度）
- [x] 推理关键指标（TTFT / ITL / TPS / QPS / MFU）
- [x] SGLang（RadixAttention、与 vLLM 对比）

### 阶段 3 广度补充（全部完成）
- [x] 模型量化（INT8/INT4、为什么加速 Decode、精度 vs 速度）
- [x] NCCL 基础（AllReduce / AllGather、GPU 间通信）
- [x] Tensor Parallelism（按行/列切分、与 KV Cache 关系）
- [x] MoE（Router + Expert 选择、DeepSeek-V3 例子）
- [x] ONNX / TensorRT（模型交换格式、推理优化引擎、层融合）
- [x] Speculative Decoding（小模型草拟 + 大模型验证）
- [x] 推理系统对比（vLLM vs SGLang vs TensorRT-LLM）

### 额外收获
- [x] 数据类型详解（FP16/BF16/TF32/INT8/INT4 区别）
- [x] NVLink vs InfiniBand（同机高带宽 vs 跨机网络）
- [x] Hugging Face 是什么（AI 界的 GitHub + Transformers 库）
- [x] 岗位方向分析和个人定位
- [x] nano-vllm 项目选定为下一步实践方向
- [x] CUTLASS 学习评估（现阶段不需要）
- [x] AI 辅助写代码的方法论

---

## 💡 核心理解

### 1. vLLM 三大组件

```
Scheduler（调度器）→ 决定每一步处理哪些请求
  维护三个队列: Waiting（等待）、Running（运行）、Swapped（换出）
  显存不足时可以抢占：把请求的 KV Cache 从 GPU 换到 CPU

BlockManager（块管理器）→ 管理 Block 的分配和释放
  维护 Free List、Block Table、处理 Copy-on-Write
  是 PagedAttention 的具体实现

Worker（执行器）→ 在 GPU 上执行实际计算
  接收 Scheduler 指令，执行模型前向计算，返回新 token
```

### 2. Continuous Batching

Static Batching：以 batch 为单位调度，整个 batch 完成才换下一个 → 短请求等长请求，GPU 空转。

Continuous Batching：以 iteration 为单位调度，每一步都可以移出完成的请求、加入新请求 → GPU 始终满载。

需要 PagedAttention 配合：完成的请求释放 Block → 立刻分配给新请求 → 高效的显存周转。

### 3. 推理关键指标

| 指标 | 含义 | 对应阶段 | 优化方向 |
|------|------|---------|---------|
| TTFT | 首 token 延迟 | Prefill | FlashAttention、Prefix Caching |
| ITL | token 间延迟 | Decode | 量化、Speculative Decoding |
| TPS | 每秒生成 token 数 | 系统级 | Batching、PagedAttention |
| QPS | 每秒完成请求数 | 系统级 | TPS / 平均生成长度 |
| MFU | GPU 利用率 | 全局 | Batching 提高利用率 |

### 4. SGLang vs vLLM

vLLM 以 PagedAttention 做 Block 级内存管理，社区最大。SGLang 以 RadixAttention 做前缀级 KV Cache 共享（Radix Tree），多轮对话和 Agent 场景更优。

### 5. 广度主题速查

**量化**：把 fp16 参数压缩到 INT8/INT4，减少 HBM 读取量，加速 Memory Bound 的 Decode 阶段。核心 trade-off: 精度 vs 速度。

**NCCL**：NVIDIA 多 GPU 通信库，AllReduce 把各 GPU 的部分结果求和合并，AllGather 收集所有 GPU 的数据。类似 HPC 的 MPI。

**Tensor Parallelism**：把权重矩阵按列/行切分到多 GPU，各算一部分后 AllReduce 合并。Attention 按 head 切分。通常只在同机 NVLink 内使用（带宽 600 GB/s），跨机用 Pipeline Parallelism。

**MoE**：Router 让每个 token 只激活 top-k 个 Expert（FFN），实现大模型容量 + 小模型计算量。DeepSeek-V3: 671B 总参数，每 token 只激活 37B。

**ONNX / TensorRT**：ONNX 是通用模型交换格式（"模型的 PDF"）。TensorRT 是 NVIDIA 推理优化引擎，做层融合、精度校准、kernel 自动调优。TensorRT-LLM 是 LLM 专用版，和 vLLM 竞争。

**Speculative Decoding**：小模型快速猜多个 token，大模型一次性验证（Prefill 模式），猜对率高时加速 2-3×。

### 6. 数据类型补充

```
FP32: 32 bit 浮点，精度最高，训练常用
FP16: 16 bit 浮点（5指数+10尾数），推理常用，范围小
BF16: 16 bit 浮点（8指数+7尾数），范围和 FP32 一样大，训练不容易溢出
TF32: 19 bit，NVIDIA Ampere 专用格式
INT8: 8 bit 整数，-128~127
INT4: 4 bit 整数，-8~7
```

BF16 vs FP16：BF16 精度略低但范围大（不容易溢出），训练更稳定。Google 发明（Brain Float）。

### 7. NVLink vs InfiniBand

```
NVLink:      同机 GPU 间直连，600 GB/s，适合频繁通信的 Tensor Parallelism
InfiniBand:  跨机网络，25-50 GB/s，适合通信不频繁的 Pipeline Parallelism
```

---

## 🆚 对比总结

### Static Batching vs Continuous Batching

| | Static | Continuous |
|--|--------|-----------|
| 调度粒度 | Batch 级 | Iteration 级 |
| 短请求 | 必须等长请求完成 | 完成即释放 |
| GPU 利用率 | 低（空转严重） | 高（始终满载） |
| 实现复杂度 | 简单 | 需要 Scheduler + BlockManager |

### vLLM vs SGLang vs TensorRT-LLM

| | vLLM | SGLang | TensorRT-LLM |
|--|------|--------|-------------|
| 核心技术 | PagedAttention | RadixAttention | TensorRT 优化 |
| 优势 | 社区大、通用 | 前缀复用、多轮对话 | 极致性能 |
| 劣势 | 某些场景非最优 | 社区较小 | 灵活性低、NVIDIA 绑定 |

---

## 🎯 项目决策

### 选定：nano-vllm 复现

nano-vllm 是一个约 1200 行 Python 的轻量 vLLM 实现，包含 PagedAttention、Continuous Batching、Prefix Caching、Tensor Parallelism 等核心机制，性能接近完整 vLLM。

选择理由：
- 代码量适中（1200 行 vs vLLM 几万行）
- 覆盖所有学过的推理系统概念
- 偏推理框架方向，和目标岗位匹配
- 有 benchmark 可以对比 vLLM

执行计划：
- 第一阶段：读源码，理解每个模块（3-4 天）
- 第二阶段：自己设计 + AI 辅助实现，关键函数手写（4-5 天）
- 第三阶段：跑 benchmark、写文档、推 GitHub（2-3 天）

### CUTLASS：现阶段不学

CUTLASS 学习曲线极陡（3.x 模板元编程非常复杂），对实习面试帮助不大。面试考的是 GEMM 优化原理（已掌握），不是 CUTLASS 模板用法。等入职后有实际需求再学。

---

## 🤔 方法论收获

### AI 辅助写代码的正确姿势

```
读源码:    用于已有的优质项目，逐模块读懂设计决策
AI 辅助:   自己设计架构和接口，AI 帮填实现细节，然后逐行读懂
纯手写:    核心逻辑和面试可能考的函数，关掉 AI 自己写
```

核心标准不是"代码谁写的"，而是"能不能解释清楚代码里的每个决策"。

### 个人定位

```
方向: GPU Kernel 优化 + 推理系统（偏推理框架）
优势: HPC 背景（MPI/并行计算）+ CUDA 实战 + 推理系统全栈理解
项目线: GEMM（kernel）→ FlashAttention（算子）→ nano-vllm（系统）
        从底层到上层的完整叙事
```

---

## 📊 阶段完成度

```
阶段 1: Transformer 推理基础    ✅ 完成（Day 1-2）
阶段 2: KV Cache + PagedAttention ✅ 完成（Day 3-4）
阶段 3: 广度补充               ✅ 完成（Day 4）
阶段 3.5: 实践项目             🔜 选定 nano-vllm，明天开始

比原计划提前约 2 周完成阶段 1-3！
```

---

## 🎯 Day 5+ 计划

开始 nano-vllm 项目：
1. 先读源码，理解模块结构和设计决策
2. 画出模块关系图，标注对应的概念（PagedAttention、Scheduler 等）
3. 然后开始自己动手实现

---

## 💭 个人反思

今天一天把阶段 2 收尾和阶段 3 全部搞定了。阶段 3 的广度内容确实不难，每个主题花十几分钟就能理解核心概念。但正如之前担心的——这些都是"知道"层面的知识，没有"做过"的实感。

所以决定做 nano-vllm 项目是对的。这个项目能把之前学的所有概念（KV Cache、PagedAttention、Continuous Batching、Scheduler）变成实际代码，补上"理论到实践"的最后一环。

关于 AI 辅助写代码的讨论也让我想清楚了：重要的不是代码是谁写的，而是我能不能解释清楚每个设计决策。做 nano-vllm 时要注意这一点——不是跑通就行，而是要理解每一行为什么这样写。
