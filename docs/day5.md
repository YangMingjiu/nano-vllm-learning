# Day 5 - nano-vllm 源码阅读（一）：最外层流程

> 📅 日期：2025年3月17日
> 📍 阶段：阶段 3.5 - nano-vllm 项目
> ⏱️ 学习时长：约 1h

---

## ✅ 今日完成

- [x] nano-vllm 项目结构梳理（1358 行，15 个文件）
- [x] 所有文件和之前学过概念的对应关系
- [x] `generate()` 函数逐行理解
- [x] `step()` 函数逐行理解
- [x] `schedule()` 函数逐行理解（Prefill 优先 + Decode + Preemption）
- [x] Python 语法补充（类型注解、列表推导式、三元表达式、tuple 解包、isinstance、zip、while-else）
- [x] 用具体 3 个请求的例子走完 generate + step 的完整流程

---

## 💡 核心理解

### 1. 项目结构

```
nanovllm/                          总共 1358 行
├── llm.py              (5行)      入口
├── config.py            (26行)    配置
├── sampling_params.py   (11行)    采样参数
│
├── engine/                        ← 今天重点看的部分
│   ├── sequence.py      (83行)    请求的数据结构（含 block_table）
│   ├── block_manager.py (112行)   Block 分配/释放 + Prefix Caching
│   ├── scheduler.py     (71行)    调度器（Waiting/Running 队列）   ✅ 今天读了
│   ├── llm_engine.py    (93行)    主循环（generate → step）       ✅ 今天读了
│   └── model_runner.py  (251行)   GPU 执行 + CUDA Graph
│
├── layers/                        ← 模型组件（对应 Day 1-2 学的）
│   ├── attention.py     (75行)    Attention + KV Cache 存取
│   ├── rotary_embedding.py (61行) RoPE
│   ├── layernorm.py     (50行)    RMSNorm
│   ├── linear.py        (153行)   线性层 + Tensor Parallelism
│   ├── activation.py    (14行)    SiluAndMul（SwiGLU）
│   ├── embed_head.py    (66行)    Embedding + LM Head
│   └── sampler.py       (15行)    采样
│
├── models/
│   └── qwen3.py         (215行)   Qwen3 模型定义
│
└── utils/
    ├── context.py       (27行)    全局上下文
    └── loader.py        (28行)    权重加载
```

### 2. 核心调用链

```
LLM.generate(prompts)
  │
  ├→ add_request() × N        ← 所有请求加入 waiting（在循环之前）
  │
  └→ while not finished:       ← 循环 step 直到全部完成
       step()
         ├→ scheduler.schedule()         调度：Prefill 还是 Decode？
         ├→ model_runner.run()           GPU 执行，返回新 token
         └→ scheduler.postprocess()      追加 token，检查是否完成
```

### 3. generate() 核心逻辑

去掉进度条后只有 10 行核心代码：

```
① 所有请求加入 waiting 队列
② 循环 step()，每步收集已完成的请求
③ 全部完成后，按提交顺序排序，解码成文字，返回
```

- `outputs` 用字典 `{seq_id: token_ids}` 存结果，最后按 seq_id 排序
- 排序是因为请求完成顺序不等于提交顺序（短请求先完成）

### 4. step() 核心逻辑

```
step() = 推理引擎的一次心跳

① seqs, is_prefill = scheduler.schedule()    问调度器该干什么
② token_ids = model_runner.run(seqs)          GPU 执行，每个请求生成 1 个 token
③ scheduler.postprocess(seqs, token_ids)      追加 token，检查完成
④ 返回：已完成的请求列表 + 处理的 token 数
```

- `num_tokens` 正数表示 Prefill（总 token 数），负数表示 Decode（请求数取负）
- 大多数 step 可能没有请求完成，output 是空列表

### 5. schedule() 核心逻辑

```
schedule() 做的事:

  先尝试 Prefill:
    while waiting 不为空 and 没选够:
      检查 token 总数不超限 + Block 够用
      → 通过: 分配 Block，从 waiting 移到 running
      → 不通过: break
    if 选到了 → return (seqs, True)     ← Prefill 优先！

  没有新请求 → 做 Decode:
    while running 不为空:
      检查能不能追加 Block
      → 不能: 踢人（preempt）腾空间
      → 能: 加入调度列表
    return (seqs, False)
```

关键设计决策：
- **Prefill 优先**：有新请求就做 Prefill，不做 Decode
- **Preemption（抢占）**：显存不够时踢最后加入的请求（LIFO），释放它的 Block
- 被踢的请求放回 waiting 队首，下次有机会再处理
- 没有 Swapped 队列（简化版，不做 GPU↔CPU 换出换入）

### 6. Continuous Batching 在哪里体现

在 generate() 的 while 循环中：
- 请求完成 → postprocess 中标记 FINISHED → 释放 Block → 从 running 移出
- 下一次 schedule() 时，running 里少了一个请求，腾出的 Block 可以给新请求用
- 如果有新请求在 waiting 里 → 立刻做 Prefill 加入

generate() 最后等全部完成才 return，这是因为它是离线批处理接口。但**引擎内部是 Continuous Batching 的**——完成一个释放一个，不会等其他请求。

### 7. Batching 的意义

多个请求打包成 batch 一起丢给 GPU，不是为了"同时服务"，核心是**提高 GPU 利用率**：

```
Decode 时每步都要读 14GB 参数（Memory Bound）

1 个请求:  读 14GB → 算 1 个 token  → GPU 利用率极低
32 个请求: 读 14GB → 算 32 个 token → 同样 IO，32 倍产出
```

---

## 🆚 文件和学过概念的对应关系

```
学过的概念               →  nano-vllm 中的位置
──────────────────────────────────────────────────
Embedding 查表           →  layers/embed_head.py + qwen3.py
RMSNorm                 →  layers/layernorm.py
MHA 5 步流程            →  qwen3.py: Qwen3Attention.forward()
  QKV 合并投影          →  qkv_proj（一次 GEMM 算出 Q,K,V）
  分头                  →  q.view(-1, num_heads, head_dim)
  RoPE                  →  layers/rotary_embedding.py
  Attention + KV Cache  →  layers/attention.py（调用 flash_attn）
  W_O 输出投影          →  o_proj
SwiGLU FFN              →  qwen3.py: Qwen3MLP
Residual Connection     →  qwen3.py: DecoderLayer 的 residual 传递
LM Head                 →  qwen3.py: lm_head（+ weight tying）

KV Cache 分配           →  model_runner.py: allocate_kv_cache()
Block / Block Table     →  block_manager.py + sequence.py
PagedAttention          →  block_manager.py + attention.py 的 block_table
Prefix Caching          →  block_manager.py: hash_to_block_id
Scheduler 队列          →  scheduler.py: waiting + running
Continuous Batching     →  llm_engine.py: while + step() 循环
Prefill vs Decode       →  scheduler.py: schedule() 返回 is_prefill
Preemption              →  scheduler.py: preempt()
Tensor Parallelism      →  layers/linear.py: ColumnParallel / RowParallel
CUDA Graph              →  model_runner.py: capture_cudagraph()
```

---

## 📝 Python 语法收获

| 语法 | 说明 | 例子 |
|------|------|------|
| 类型注解 | 冒号后面标注类型，不影响运行 | `prompts: list[str]` |
| `A \| B` | 类型注解中的"或" | `list[str] \| list[int]` |
| `-> type` | 返回值类型注解 | `-> list[str]` |
| 默认参数 | 参数有默认值，调用时可不传 | `use_tqdm: bool = True` |
| tuple 解包 | 多个值用逗号赋给多个变量 | `a, b = func()` |
| 返回多个值 | Python 函数可以 return 多个值（本质是 tuple） | `return seqs, True` |
| 列表推导式 | 一行写 for + if 生成列表 | `[x for x in list if cond]` |
| 三元表达式 | 一行写 if-else | `A if cond else B` |
| isinstance | 检查变量类型 | `isinstance(x, list)` |
| zip | 多个列表配对遍历 | `for a, b in zip(l1, l2)` |
| while-else | while 正常结束（没 break）执行 else | `while ...: ... else: ...` |
| `{}` 字典 | key-value 存储 | `outputs = {1: [450], 0: [278]}` |
| `deque` | 双端队列，支持 popleft | `from collections import deque` |
| 连续赋值 | 多变量赋同一个值 | `a = b = 0.` |

---

## 🤔 今日反思

今天是第一次读多文件的系统级项目，一开始确实有点摸不着头脑——之前的项目都是单个 kernel 文件，逻辑线性；nano-vllm 有十几个文件互相调用，每看一个函数就想钻进去看它调用的下一层。

关键的思维转变是**先看宏观，强制只看一层**。今天只看了最外面的 generate() → step() → schedule() 这三个函数，加起来不到 60 行核心代码，就把整个系统"做什么"搞清楚了。细节（Block 怎么分配、模型怎么跑）留到后面再看。

Python 语法也补了不少。类型注解、列表推导式这些一开始看着很吓人，但拆开后都不复杂。后面读代码应该会越来越顺。

用具体例子走一遍 3 个请求的完整生命周期后，generate 和 step 的关系就清楚了——generate 是老板（分配任务 + 收集结果），step 是员工（每次干一步活）。schedule 是调度员（决定这一步干什么）。

---

## 🎯 Day 6 计划

往下挖一层，看 Sequence 和 BlockManager 的细节：
1. `sequence.py` → block_table、状态管理、token 追加
2. `block_manager.py` → Block 分配/释放、Prefix Caching 的哈希逻辑
3. 如果时间够 → `attention.py`（KV Cache 存取 + flash_attn 调用）
