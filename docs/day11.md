# Day 11 - Benchmark 实验 + 统计功能开发 + 图表可视化

> 📅 日期：2025年3月23日
> 📍 阶段：阶段 3.5 - nano-vllm 项目收尾
> ⏱️ 学习时长：约 2h

---

## ✅ 今日完成

- [x] 不同 batch size 的 benchmark 实验（9 组：1, 2, 4, 8, 16, 32, 64, 128, 256）
- [x] 给 scheduler.py 添加统计计数器（num_prefill / num_decode / num_preempt）
- [x] 给 block_manager.py 添加 Prefix Cache 命中计数（num_cache_hit）
- [x] 给 scheduler.py 添加 reset_stats() 方法
- [x] 修改 llm_engine.py 打印统计结果
- [x] 编写 bench_batch.py（带统计数据收集的 benchmark 脚本）
- [x] 编写 plot_bench.py（三图可视化）
- [x] 生成最终图表：Throughput + Per-Token Latency + Preemptions

---

## 💡 核心发现

### Benchmark 数据

```
 Batch |    Tok/s | Prefill |  Decode | Preempt | Cache Hit
------------------------------------------------------------
     1 |      161 |       1 |     303 |       0 |         0
     2 |      235 |       1 |     504 |       0 |         1
     4 |      430 |       1 |     435 |       0 |         2
     8 |      671 |       1 |     500 |       0 |         4
    16 |     1058 |       1 |     511 |       0 |         7
    32 |     1727 |       1 |     499 |       0 |        13
    64 |     2507 |       1 |     507 |       0 |        23    ← peak!
   128 |     1308 |      62 |     890 |      64 |        38    ← preemption 爆发!
   256 |     1222 |     164 |    1434 |     145 |        41
```

### 关键洞察

```
1. Batching 的收益巨大
   batch 1→64: 吞吐量 161→2507 tok/s（16× 提升！）
   每 token 耗时: 6.22ms→0.40ms（16× 降低）
   原因: 多请求共享读模型参数的 IO 成本（Decode 阶段是 Memory Bound）

2. 显存是瓶颈，不是算力
   batch 1-64: Preemption = 0，Prefill 只需 1 步 → 显存够用
   batch 128:  Preemption = 64！Prefill 变成 62 步！→ 显存不够了
   batch 256:  Preemption = 145！Prefill 变成 164 步！→ 更严重

3. Preemption 导致吞吐量下降
   被踢的请求要重新 Prefill → 浪费 GPU 时间
   batch 128/256 的大量 Prefill steps 就是重复计算的证据
   → 8GB RTX 3070 的甜点是 ~64 并发请求

4. Prefix Cache 在起作用
   Cache Hit 随 batch size 增大 → 相同前缀的请求越多，命中越多
   但在这个实验中效果有限（随机 token，前缀重合不多）
```

### 三张图的故事

```
左图 (Throughput):       Batching 让吞吐量从 161 提升到 2507（16×）
中图 (Per-Token Latency): 每 token 成本从 6.22ms 降到 0.40ms
右图 (Preemptions):      超过 64 并发后，显存不够触发频繁 Preemption，性能反而下降
```

---

## 🔧 代码修改记录

### scheduler.py 新增内容

```python
# __init__ 中新增计数器
self.num_prefill = 0
self.num_decode = 0
self.num_preempt = 0

# schedule() 中新增计数
if scheduled_seqs:
    self.num_prefill += 1        # Prefill step 计数
    return scheduled_seqs, True

self.num_decode += 1             # Decode step 计数
self.running.extendleft(...)
return scheduled_seqs, False

# preempt() 中新增计数
def preempt(self, seq):
    self.num_preempt += 1        # Preemption 计数
    ...

# 新增 reset 方法
def reset_stats(self):
    self.num_prefill = 0
    self.num_decode = 0
    self.num_preempt = 0
    self.block_manager.num_cache_hit = 0
```

### block_manager.py 新增内容

```python
# __init__ 中新增
self.num_cache_hit = 0

# allocate() 中 cache 命中时计数
else:  # cache 命中
    self.num_cache_hit += 1
    seq.num_cached_tokens += self.block_size
```

### llm_engine.py 新增内容

```python
# generate() 末尾打印统计
print(f"\n=== Scheduler Stats ===")
print(f"Prefill steps:  {self.scheduler.num_prefill}")
print(f"Decode steps:   {self.scheduler.num_decode}")
print(f"Preemptions:    {self.scheduler.num_preempt}")
print(f"Cache hits:     {self.scheduler.block_manager.num_cache_hit}")
```

---

## 📝 Python 语法收获

| 语法 | 说明 | 例子 |
|------|------|------|
| `self.x += 1` vs `x += 1` | 前者修改对象属性，后者是局部变量 | 忘加 self 会报 NameError |
| `r.get("key", default)` | 字典取值，不存在时返回默认值 | `r.get("preemptions", 0)` |
| f-string 格式化 | `f"{value:>8}"` 右对齐 8 格 | `f"{r['throughput']:>8}"` |

---

## 📊 最终产出

```
文件:
  bench_batch.py       → 带统计数据收集的 benchmark 脚本
  plot_bench.py        → 三图可视化脚本
  bench_results.json   → 原始实验数据
  bench_throughput.png  → 最终图表

代码修改:
  scheduler.py         → +3 个计数器 + reset_stats()
  block_manager.py     → +1 个计数器
  llm_engine.py        → +统计打印
```

---

## 📊 nano-vllm 项目总成果

```
源码阅读:      ✅ 1358 行全部读懂
中文注释:      ✅ scheduler / block_manager / llm_engine / attention
新功能:        ✅ 调度统计（Prefill/Decode/Preempt/CacheHit 计数）
Benchmark:     ✅ 9 组 batch size 实验 + 三图可视化
运行环境:      ✅ RTX 3070 Laptop + Qwen3-0.6B
关键发现:      ✅ Batching 16× 提升 + Preemption 导致性能下降

面试可讲的点:
  "我读了 nano-vllm 的 1358 行源码，理解了 Scheduler、BlockManager、
   Attention 的完整流程。还自己加了调度统计功能，通过 benchmark 发现
   batch=64 时吞吐量达到峰值 2507 tok/s，但 batch=128+ 时因为
   8GB 显存不足触发频繁 Preemption（145 次），导致吞吐量下降到 1222。
   这验证了 PagedAttention 的内存管理对推理性能的关键影响。"
```

---

## 🗺️ 整体进度

```
阶段 1: Transformer 推理基础       ✅ Day 1-2
阶段 2: KV Cache + PagedAttention  ✅ Day 3-4
阶段 3: 广度补充                   ✅ Day 4
阶段 3.5: nano-vllm 项目          ✅ Day 5-11（完成！）

阶段 4: 整合 + 项目包装            🔜 下一步
阶段 5: 八股文 + 简历              🔜
阶段 6: 面试冲刺                   🔜

当前: 3月23日
→ 比原计划提前约 2 周进入阶段 4
```

---

## 💭 今日反思

今天是 nano-vllm 项目最有成就感的一天。

最大的收获是 benchmark 数据直接验证了之前学过的理论：
- Day 4 学的 "Batching 提高 GPU 利用率" → 数据显示 16× 吞吐量提升
- Day 3 学的 "KV Cache 是推理显存的最大消耗者" → 128 并发时显存爆了
- Day 5 学的 "Preemption 把请求踢回 waiting" → 145 次 Preemption 直接导致性能下降
- Day 6 学的 "deallocate 释放 Block" → 被踢的请求 Block 释放后又要重新 Prefill

理论和实践终于对上了。

另外发现了一个有趣的现象：两次跑 benchmark 的峰值不在同一个 batch size（第一次 32，第二次 64）。第二次更准确因为计数器每次重置了。这也是工程中常见的问题——benchmark 的条件要严格控制，否则结果不可复现。

---

## 🎯 下一步

进入阶段 4：整合 + 项目包装
1. 整理 nano-vllm 成果推 GitHub（新建 repo）
2. 写 3 分钟技术叙事
3. 打磨三个项目的 GitHub README
4. 写简历初稿
