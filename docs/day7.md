# Day 7 - nano-vllm 源码阅读（三）：Attention + ModelRunner

> 📅 日期：2025年3月19日
> 📍 阶段：阶段 3.5 - nano-vllm 项目
> ⏱️ 学习时长：约 3h

---

## ✅ 今日完成

- [x] `attention.py`（75行）：store_kvcache Triton kernel + Attention forward
- [x] `model_runner.py` 核心部分：allocate_kv_cache / prepare_prefill / prepare_decode / run
- [x] CUDA Graph 概念理解（录制 + 回放省 CPU 调度开销）
- [x] varlen 概念回顾（不同长度序列拼一起，cu_seqlens 标记边界）
- [x] 完整数据流梳理（prepare → 全局 Context → Attention 层取用）

---

## 💡 核心理解

### 1. attention.py 做两件事

**第一件：store_kvcache（Triton kernel）**

把新 token 的 K 和 V 写到 KV Cache 的正确物理位置：

```
slot_mapping 告诉 kernel 写到哪:
  slot = 物理块编号 × block_size + 块内偏移

  比如 token 属于物理块 3，是块内第 5 个:
  slot = 3 × 256 + 5 = 773
  → 把 K 和 V 写到 cache 的第 773 个位置
```

Triton 语法和 CUDA 的对应关系：
```
CUDA:   int idx = blockIdx.x;          Triton: idx = tl.program_id(0)
CUDA:   float val = key[offset];       Triton: val = tl.load(key_ptr + offset)
CUDA:   cache[slot] = val;             Triton: tl.store(cache_ptr + slot, val)
```

**第二件：Attention.forward**

根据是 Prefill 还是 Decode，调用不同的 flash_attn 函数：

```
Prefill → flash_attn_varlen_func
  多序列拼成一行，cu_seqlens 标记边界
  完整序列的 Q×K^T → Softmax → ×V

Decode → flash_attn_with_kvcache
  每个请求只有 1 个 query
  query 对 paged KV Cache 做 Attention
  block_table 告诉它 KV 散落在哪些物理块
```

nano-vllm 没有自己写 Attention kernel，而是调用 flash_attn 库，自己只写了 store_kvcache 的 Triton kernel。

### 2. varlen 是什么

不同长度的序列**不 padding，直接拼成一行**，用 cu_seqlens 标记边界：

```
传统 padding:
  seq_0: [a, b, c, d, e, f]     6 个
  seq_1: [g, h, 0, 0, 0, 0]     补了 4 个无用的 padding
  seq_2: [i, j, k, l, 0, 0]     补了 2 个无用的 padding
  → 有浪费

varlen:
  input: [a, b, c, d, e, f, g, h, i, j, k, l]    拼一行，零浪费
  cu_seqlens = [0, 6, 8, 12]                       标记边界
  → seq_0 = input[0:6], seq_1 = input[6:8], seq_2 = input[8:12]
```

每个序列的 Attention 是独立的——seq_0 的 token 不会 attend to seq_1。cu_seqlens 告诉 flash_attn 在哪里切分。

varlen 不是新算法，只是 FlashAttention 的数据打包方式。我的 FA 项目实现的是基础版（同长度 batch），varlen 是工程优化版（变长 batch）。核心 Attention 计算完全一样。

### 3. model_runner.py 的核心功能

**allocate_kv_cache：一次性分配所有 KV Cache 显存**

```python
block_bytes = 2 × num_layers × block_size × num_kv_heads × head_dim × dtype_bytes
#             ^K和V  ^层数      ^每Block token  ^头数        ^头维度    ^数据类型

num_blocks = 剩余显存 // block_bytes

kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
#                       ^     ^          ^           ^            ^            ^
#                      K/V   层数      Block数    每Block token   KV头数      头维度
```

分配后把每一层的 cache 分给对应的 Attention 模块：
```
层 0: k_cache = kv_cache[0, 0], v_cache = kv_cache[1, 0]
层 1: k_cache = kv_cache[0, 1], v_cache = kv_cache[1, 1]
...
```

**prepare_prefill vs prepare_decode**

两者本质做同一件事：把 Sequence 对象转换成 GPU 能吃的 tensor。区别在于数据量：

| | prepare_prefill | prepare_decode |
|--|----------------|---------------|
| input_ids | 所有 token 拼一行 | 每个 seq 只取 last_token |
| positions | 每个 seq 从 0 开始 | 每个 seq 的最后位置 |
| 序列边界 | cu_seqlens | 不需要 |
| Cache 长度 | 不需要 | context_lens |
| slot_mapping | 所有 token 的物理位置 | 每个 seq 1 个物理位置 |
| block_tables | 有 prefix cache 时才需要 | 始终需要 |

**CUDA Graph：Decode 时的性能优化**

```
没有 CUDA Graph:
  每次 Decode: CPU 逐个调度几百个 kernel → ~几 ms 的 CPU 开销

有 CUDA Graph:
  第一次: "录制"所有 kernel 调用
  之后: 一次 CPU 调用"回放" graph → 几乎零 CPU 开销

Prefill 不用 CUDA Graph（输入长度每次不同，无法预录制）
Decode 用（每步结构相同，预录制多个 batch size 的 graph）
```

### 4. 完整数据流

```
prepare_prefill/decode
    │
    ├→ 返回 input_ids, positions
    │   → 直接传给 model(input_ids, positions)
    │   → Embedding → 32 层 Transformer → LM Head → logits
    │
    └→ 存入全局 Context（set_context）
        → slot_mapping     → store_kvcache 用（写 KV 到哪）
        → cu_seqlens       → flash_attn_varlen_func 用（序列边界）
        → context_lens     → flash_attn_with_kvcache 用（Cache 长度）
        → block_tables     → flash_attn 用（KV 在哪些物理块）

Attention 层通过 get_context() 取出这些值使用
```

为什么用全局 Context？因为 slot_mapping 等信息需要传到模型内部深层的 Attention 里，一层层传参数太啰嗦。存到全局变量，Attention 层直接取，代码更简洁。

---

## 📝 新学的语法/概念

| 语法/概念 | 说明 |
|-----------|------|
| Triton kernel | 用 Python 语法写 GPU kernel，和 CUDA 等价但更简洁 |
| `@triton.jit` | 装饰器，标记函数为 Triton kernel |
| `tl.program_id(0)` | 等价于 CUDA 的 `blockIdx.x` |
| `tl.load / tl.store` | 等价于 CUDA 的全局内存读写 |
| `[(N,)]` | Triton 的 grid 大小，类似 CUDA 的 `<<<grid, block>>>` |
| `torch.cuda.mem_get_info()` | 获取 GPU 剩余显存和总显存 |
| `torch.cuda.CUDAGraph()` | CUDA Graph 对象，录制和回放 kernel 序列 |
| `getattr(obj, "name", default)` | 获取对象属性，不存在则返回默认值 |
| `.numel()` | tensor 中元素的总数 |
| `.unsqueeze(1)` | 在指定位置插入一个维度 |
| 全局 Context 模式 | 用全局变量传递信息，避免层层传参 |

---

## 📊 源码阅读进度

```
engine/ 目录（核心引擎）:
  [x] llm_engine.py    (93行)    Day 5 ✅
  [x] scheduler.py     (71行)    Day 5 ✅
  [x] sequence.py      (83行)    Day 6 ✅
  [x] block_manager.py (112行)   Day 6 ✅
  [x] model_runner.py  (251行)   Day 7 ✅  ← 今天

layers/ 目录（模型组件）:
  [x] attention.py     (75行)    Day 7 ✅  ← 今天
  [ ] rotary_embedding.py (61行) Day 8 🔜
  [ ] layernorm.py     (50行)    Day 8 🔜
  [ ] linear.py        (153行)   Day 8 🔜
  [ ] activation.py    (14行)    Day 8 🔜
  [ ] embed_head.py    (66行)    Day 8 🔜
  [ ] sampler.py       (15行)    Day 8 🔜

models/ 目录:
  [ ] qwen3.py         (215行)   Day 8 🔜

utils/ 目录:
  [x] context.py       (27行)    Day 7 ✅（通过数据流分析理解了）
  [ ] loader.py        (28行)    不重要

进度: 712 / 1358 行 ≈ 52%
  engine/ 完成 ✅
  attention.py 完成 ✅
  剩余: layers/ 其他 + qwen3.py（都是 Transformer 组件，应该很快）
```

---

## 🤔 今日反思

今天看完了 attention.py 和 model_runner.py 的核心部分。这两个文件比 Day 5-6 看的引擎逻辑要难一些——涉及 Triton kernel 语法、flash_attn 库的 API、以及 GPU 显存管理的具体数值计算。

一个重要的收获是搞清楚了**数据流**。之前看 prepare_prefill 和 prepare_decode 时觉得它们在做一堆看不懂的数据处理，不知道谁在用这些数据。搞清楚全局 Context 机制后恍然大悟——prepare 把 slot_mapping、cu_seqlens 等存到全局 Context，然后模型内部的 Attention 层通过 get_context() 取出来用。

连续三天读新代码，有种"每个函数单独看能理解，但串起来就模糊"的感觉。这是正常的——信息量太大，大脑需要时间整理。明天计划先花上午回顾 Day 5-7 的笔记，画一张完整的调用链图，让知识沉淀一下。下午如果踏实了再看 qwen3.py 和 layers。

另外意识到一个读代码的方法问题：不应该试图读懂每一行语法。像 flash_attn_varlen_func 这种库函数，只需要知道"它做 Prefill 的 Attention，输入是 QKV + cu_seqlens，输出是 Attention 结果"就够了。把库调用当黑盒，把精力放在理解系统的调用链和数据流上。

---

## 🎯 Day 8 计划

上午: 回顾 Day 5-7 笔记，画完整调用链图，巩固已学内容
下午: qwen3.py + layers/（Transformer 组件的代码版，应该轻松很多）
