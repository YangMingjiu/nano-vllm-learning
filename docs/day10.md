# Day 10 - attention.py 注释 + 环境搭建 + 首次运行成功

> 📅 日期：2025年3月22日
> 📍 阶段：阶段 3.5 - nano-vllm 项目
> ⏱️ 学习时长：约 5h（含环境搭建折腾时间）

---

## ✅ 今日完成

- [x] `attention.py` 中文注释完成
- [x] 理解模型 vs 推理引擎的关系
- [x] 本地环境搭建：Miniconda + conda 虚拟环境
- [x] 依赖安装：PyTorch 2.6.0 + flash-attn 2.7.3 + transformers + triton
- [x] 解决 flash-attn 编译问题（CUDA 版本不匹配、WSL 内存）
- [x] 解决 rope_scaling 兼容性 bug
- [x] **nano-vllm 首次运行成功！** 🎉

---

## 💡 核心理解

### 1. 模型 vs 推理引擎

```
模型 (Qwen3-0.6B) = 厨师（会做菜）
  → 下载下来就是一堆权重参数（W_Q, W_K, W_V 等）
  → 用 HuggingFace Transformers 几行代码就能跑
  → 但只适合 1 个用户慢慢用

推理引擎 (nano-vllm) = 餐厅后厨系统（排单、调度、并行出菜）
  → 不改变厨师的手艺（模型能力一样）
  → 但能同时高效服务几百个用户
  → 通过 PagedAttention + Continuous Batching + CUDA Graph 等优化

HuggingFace 直接跑 vs nano-vllm:
  → 没有 KV Cache 管理 → 显存浪费
  → 没有 PagedAttention → 并发上不去
  → 没有 Continuous Batching → GPU 空转
  → vLLM 比 HuggingFace 快 8-24×
```

### 2. attention.py 完整注释

```python
## Triton kernel：把新 token 的 K, V 写到 KV Cache 的指定位置
## 每个线程块处理一个 token
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)       ## 当前处理第几个 token
    slot = tl.load(slot_mapping_ptr + idx)  ## 这个 token 写到 cache 的哪个位置
    if slot == -1: return        ## CUDA Graph padding 标记，跳过

    ## 从 key/value 张量读取这个 token 的 D 个数值
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))

    ## 写入 cache 的 slot 位置
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)


class Attention:
    def forward(self, q, k, v):
        ## ① 从全局 Context 获取 prepare 阶段存入的元数据
        context = get_context()

        ## ② 把新的 K, V 写入 KV Cache
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        ## ③ 执行 Attention
        if context.is_prefill:
            ## Prefill: 多序列打包，cu_seqlens 标记边界
            ## 有 prefix cache 时从 cache 读完整 K, V
            flash_attn_varlen_func(q, k, v, cu_seqlens=..., block_table=...)
        else:
            ## Decode: 1 个 query 对 paged KV Cache
            flash_attn_with_kvcache(q, k_cache, v_cache, block_table=...)
```

### 3. 关于 model_runner 和 attention 的理解

这两个文件和 scheduler/block_manager 不同——不是"流程逻辑"而是"数据转换"：

```
scheduler + block_manager = 流程和逻辑（用例子能串起来）
model_runner + attention   = 数据处理和 GPU 操作（把数据从一种格式转成另一种）

model_runner 的角色:
  → 把 Sequence 对象"翻译"成 GPU 能吃的 tensor
  → 把元数据存到全局 Context
  → 调用模型 forward，然后采样

attention 的角色:
  → 从 Context 取元数据
  → 把 KV 写到 cache（store_kvcache）
  → 调用 flash_attn 做计算
```

面试时对这两个文件的要求：**说清楚"做什么"就够了**，不需要解释每一行 tensor 操作。

---

## 🔧 环境搭建记录

### 最终可用的环境配置

```
OS: WSL2 Ubuntu
GPU: RTX 3070 Laptop (8GB)
CUDA (系统): 13.0
Python: 3.12 (conda)
PyTorch: 2.6.0+cu124
flash-attn: 2.7.3 (预编译 wheel)
transformers: 5.3.0
triton: 3.2.0
```

### 遇到的问题和解决方案

| 问题 | 原因 | 解决 |
|------|------|------|
| pip 报 externally-managed-environment | 系统 Python 不让直接装包 | 用 conda 创建虚拟环境 |
| conda tos 错误 | 新版 conda 需要接受条款 | `conda tos accept` |
| flash-attn import 报 undefined symbol | flash-attn 和 PyTorch 版本不匹配 | 统一版本 |
| PyTorch 被自动升级到 2.10.0 | `--force-reinstall` 拉了最新 torch | 固定 torch==2.6.0 |
| flash-attn 编译报 CUDA 版本不匹配 | 系统 CUDA 13.0 vs PyTorch CUDA 12.4 | conda 装 cuda-toolkit + export CUDA_HOME |
| WSL 频繁断开 | flash-attn 编译吃内存 | 改用预编译 wheel，不从源码编译 |
| unhashable type: 'dict' | transformers 新版的 rope_scaling 是 dict | `sed` 改为 `rope_scaling=None` |
| 缺 psutil 模块 | flash-attn 编译依赖 | `pip install psutil` |

### 关键经验

```
1. flash-attn 安装是最大的坑
   → 优先用预编译 wheel（flash-attn==2.7.3 + torch==2.6.0）
   → 避免从源码编译（慢、吃内存、版本容易不匹配）

2. 不要让 pip 自动升级 PyTorch
   → 用 --no-deps 或固定版本号
   → PyTorch 版本变了，flash-attn 就废了

3. WSL 环境下编译大型 CUDA 项目要注意内存
   → 系统只有 15GB RAM，编译 flash-attn 可能不够
   → 预编译 wheel 是最安全的选择

4. conda 环境里的 CUDA 需要手动设置
   → export CUDA_HOME=$CONDA_PREFIX
   → export PATH=$CONDA_PREFIX/bin:$PATH
```

---

## 🎉 首次运行结果

```
~/Nvllm/nano-vllm main* ❯ python example.py
Generating: 100%|████████████| 2/2 [00:14<00:00, 7.05s/it, Prefill=6tok/s, Decode=63tok/s]

Prompt: 'introduce yourself'
Completion: (模型先 <think> 思考，然后生成自我介绍)

Prompt: 'list all prime numbers within 100'
Completion: (模型先 <think> 思考质数的定义，然后开始列举)
```

性能数据：
- Prefill: 6 tok/s
- Decode: 63 tok/s
- 2 个请求共 14 秒完成
- enforce_eager=True（关闭了 CUDA Graph）

亲眼看到了之前读过的整个流程在运行：

```
generate() → add_request × 2 → 循环 step()
  → schedule(): Prefill 两个 prompt
  → model_runner: GPU 执行，store_kvcache 写入 KV Cache
  → schedule(): Decode，每步生成 1 个 token
  → postprocess: 追加 token，遇到 max_tokens 完成
  → 收集结果，解码成文字
```

---

## 📝 注释实践总成果

已完成注释的文件：
- [x] scheduler.py 完整注释（Day 8）
- [x] llm_engine.py step() + generate() 注释（Day 8）
- [x] block_manager.py allocate / deallocate / can_append / may_append 注释（Day 9）
- [x] attention.py store_kvcache + Attention.forward 注释（Day 10）

---

## 📊 项目总进度

```
源码阅读:     ✅ 100% 完成（1358 行）
中文注释:     ✅ 4 个核心文件完成
本地运行:     ✅ 首次运行成功
Benchmark:    🔜 还没跑

nano-vllm 项目阶段基本完成！
```

---

## 🗺️ 回顾整体计划

```
阶段 1: Transformer 推理基础       ✅ Day 1-2 完成
阶段 2: KV Cache + PagedAttention  ✅ Day 3-4 完成
阶段 3: 广度补充                   ✅ Day 4 完成
阶段 3.5: nano-vllm 项目          ✅ Day 5-10（今天基本完成）
  - Day 5:  最外层流程 (generate → step → schedule)
  - Day 6:  Sequence + BlockManager
  - Day 7:  attention + model_runner
  - Day 8:  回顾巩固 + qwen3/layers + 注释 scheduler/llm_engine
  - Day 9:  注释 block_manager + 完整例子
  - Day 10: 注释 attention + 环境搭建 + 首次运行成功

阶段 4: 整合 + 项目包装            🔜 下一步
阶段 5: 八股文 + 简历              🔜
阶段 6: 面试冲刺                   🔜

当前日期: 3月22日
原计划阶段 4 开始: 4月4日
→ 还有约 2 周的富余时间！
```

### 剩余时间可以做的事

```
选项 A: 继续深化 nano-vllm
  → 跑 benchmark + 和 vLLM 对比
  → 整理成完整的中文注释版推 GitHub
  → 写一篇学习博客

选项 B: 直接进入阶段 4（整合 + 项目包装）
  → 整理 3 分钟技术叙事
  → 打磨 GitHub README
  → 写简历初稿

选项 C: 两者结合
  → 花 2-3 天整理 nano-vllm 成果
  → 然后进入阶段 4
```

---

## 💭 今日反思

今天最大的成就是**亲眼看到自己读过的代码在真实运行**。之前 5 天都是在纸上谈兵（读代码、写注释、画例子），今天终于看到 Qwen3-0.6B 通过 nano-vllm 的 Scheduler、BlockManager、ModelRunner 生成了真实的文本。Prefill 6tok/s、Decode 63tok/s 这些数字不再是抽象概念，而是真实的性能数据。

环境搭建折腾了很久——flash-attn 的安装是最大的坑，涉及 CUDA 版本匹配、PyTorch 版本、WSL 内存限制等一系列问题。最终的经验是：**用预编译 wheel，不要从源码编译**。这也是工程实践中很重要的一课——不是所有东西都需要从源码编译，能用预编译包就用。

还有一个收获是理解了模型和推理引擎的关系。之前一直在学推理引擎的内部机制，今天才真正明白：模型是"厨师"，推理引擎是"后厨管理系统"。模型本身用 HuggingFace 几行代码就能跑，推理引擎解决的是"怎么高效地同时服务很多用户"的问题。
