# Day 8 - 巩固回顾 + 注释实践 + 模型代码阅读

> 📅 日期：2025年3月20日
> 📍 阶段：阶段 3.5 - nano-vllm 项目
> ⏱️ 学习时长：约 3h

---

## ✅ 今日完成

- [x] 回顾测试：6 个核心问题自测 + 纠正
- [x] waiting / running 队列机制完整回顾
- [x] scheduler.py 完整中文注释（Prefill + Decode 部分）
- [x] llm_engine.py 中文注释（step + generate）
- [x] 完整流程例子（3 个请求，含 Prefill、Decode、请求完成、Preemption、重新 Prefill）
- [x] qwen3.py 阅读（Qwen3Attention / Qwen3MLP / DecoderLayer / Model / ForCausalLM）
- [x] layers/ 目录阅读（activation / layernorm / rotary_embedding / sampler）
- [x] Gumbel-max 采样技巧（sampler.py 的实际实现）

---

## 💡 回顾测试结果

自测 6 个问题，结果：

| 问题 | 结果 | 纠正 |
|------|------|------|
| step() 做了哪三步 | ❌ 和 schedule 混了 | step = 调度 → GPU 执行 → 后处理 |
| schedule() 怎么决定 Prefill/Decode | ⭕ 大致对 | waiting 有请求 → Prefill，否则 → Decode |
| block_table 是什么 | ✅ | 列表，下标=逻辑块，值=物理块 |
| Block 怎么释放 | ⭕ 漏了 ref_count | 先 ref_count-1，降到 0 才放回 Free List |
| prepare_prefill vs decode 区别 | ✅ | Prefill 多个 token，Decode 1 个 |
| slot_mapping 干什么 | ⭕ 不够精确 | 物理块编号 × block_size + 块内偏移 = cache 中的绝对位置 |

**关键纠正：step() 和 schedule() 的职责要分清**

```
step()     = 老板，串联三个模块（调度 → 执行 → 后处理）
schedule() = 调度员，选请求 + 管 Block
model_runner = 厨房，GPU 干活
postprocess  = 记录员，追加 token + 检查完成
```

---

## 🔥 完整流程例子（最重要！）

3 个请求，block_size=4，Free List 只有 5 个 Block（显存紧张）：

```
seq_0: "What is AI?"  → 6 个 token → 需要 2 个 Block
seq_1: "Hi"           → 2 个 token → 需要 1 个 Block
seq_2: "Hello world"  → 4 个 token → 需要 1 个 Block
```

### add_request 阶段（step 之前）

```
waiting = [seq_0, seq_1, seq_2]
running = []
free = [0, 1, 2, 3, 4]
```

### Step 1: schedule() → Prefill

```
Prefill 循环:
  seq_0: 需要 2 Block, free 有 5 → 够 → allocate → block_table=[0,1]
  seq_1: 需要 1 Block, free 有 3 → 够 → allocate → block_table=[2]
  seq_2: 需要 1 Block, free 有 2 → 够 → allocate → block_table=[3]
  
  全部从 waiting 移到 running

return ([seq_0, seq_1, seq_2], True)

状态:
  waiting = []
  running = [seq_0, seq_1, seq_2]
  free = [4]  ← 只剩 1 个！
```

### Step 1: GPU Prefill → postprocess

```
GPU 给每个请求生成第 1 个 token:
  seq_0 → token 450     → append → num_tokens=7, 没完成
  seq_1 → token 2 (EOS) → append → FINISHED! → deallocate → Block 2 释放
  seq_2 → token 6324    → append → num_tokens=5, 没完成

状态:
  waiting = []
  running = [seq_0, seq_2]        ← seq_1 完成了，移出
  free = [4, 2]                   ← Block 2 回收了
  outputs = {1: [2]}              ← seq_1 的结果收集了
```

### Step 2: schedule() → Decode

```
waiting 为空 → 进入 Decode

  seq_0: popleft → running = [seq_2]
    can_append? 7%4==3 → 不需要新 Block → True
    may_append → 什么都不做
    scheduled_seqs = [seq_0]

  seq_2: popleft → running = []
    can_append? 5%4==1 → 需要 1 个新 Block → free 有 [4,2] → True
    may_append → 分配 Block 4 → block_table=[3,4]
    scheduled_seqs = [seq_0, seq_2]
    free = [2]

  extendleft → running = [seq_0, seq_2]
  return ([seq_0, seq_2], False)
```

### Step 2: GPU Decode → postprocess

```
每个请求输入 last_token，生成 1 个新 token:
  seq_0: 450 → 生成 278 → num_tokens=8, 没完成
  seq_2: 6324 → 生成 373 → num_tokens=6, 没完成

状态:
  waiting = []
  running = [seq_0, seq_2]
  free = [2]
```

### Step 3: schedule() → Decode（正常）

```
  seq_0: can_append? 8%4==0 → 不需要新 Block → True
    may_append → Block 1 满了，算 hash 注册 prefix cache
  seq_2: can_append? 6%4==2 → 不需要新 Block → True
    may_append → 什么都不做

  return ([seq_0, seq_2], False)

GPU Decode → seq_0 生成 token → num_tokens=9, seq_2 生成 token → num_tokens=7
```

### Step 4: schedule() → Decode（显存不够，触发 Preemption！）

```
  seq_0: popleft → running = [seq_2]
    can_append? 9%4==1 → 需要 1 个新 Block
    free = [2] → 有 1 个 → True
    may_append → 分配 Block 2
    scheduled_seqs = [seq_0]
    free = []  ← 空了！

  seq_2: popleft → running = []
    can_append? 7%4==3 → 不需要新 Block → True
    may_append → 什么都不做
    scheduled_seqs = [seq_0, seq_2]

  这次刚好够用，没有触发 Preemption。
```

**假设下一步 seq_0 和 seq_2 都需要新 Block，但 free 空了：**

### Step 5: schedule() → Decode（真的不够了！）

```
  seq_0: popleft → running = [seq_2]
    can_append? → 需要 1 个新 Block → free = [] → False!

    踢人循环:
      running 有 seq_2 → preempt(running.pop())
      → preempt(seq_2): 释放 seq_2 的所有 Block → 放回 waiting
      → free = [3, 4]  ← seq_2 的 Block 释放了

      can_append? → free 有 2 个 → True → 退出踢人循环

    may_append(seq_0) → 分配 1 个 Block
    scheduled_seqs = [seq_0]

  running 空了 → while 退出
  extendleft → running = [seq_0]

  return ([seq_0], False)  ← 只有 seq_0，seq_2 被踢了

状态:
  waiting = [seq_2]      ← seq_2 被踢回来了
  running = [seq_0]      ← 只剩 seq_0
```

### Step 6: schedule() → Prefill！

```
waiting 有 seq_2 → 做 Prefill！（不是 Decode）
  → seq_2 重新 allocate → 分配新 Block → 从头重新计算 KV Cache
  
return ([seq_2], True)

注意：这一步 seq_0 不做 Decode！Prefill 优先！
```

### Step 7: schedule() → Decode（恢复正常）

```
waiting 为空 → Decode
running = [seq_0, seq_2]
→ 两个都正常 Decode
→ 直到全部完成
```

### 完整时间线

```
Step  | 操作     | 处理的请求         | waiting       | running
──────┼──────────┼───────────────────┼──────────────┼─────────────────
  1   | Prefill  | [seq_0,1,2]       | []           | [seq_0,1,2]
  1后 | seq_1完成 |                   | []           | [seq_0,2]
  2   | Decode   | [seq_0, seq_2]    | []           | [seq_0,2]
  3   | Decode   | [seq_0, seq_2]    | []           | [seq_0,2]
  4   | Decode   | [seq_0, seq_2]    | []           | [seq_0,2]
  5   | Decode   | [seq_0]           | [seq_2]      | [seq_0] ←踢了seq_2
  6   | Prefill  | [seq_2]           | []           | [seq_0,2] ←seq_2回来
  7   | Decode   | [seq_0, seq_2]    | []           | [seq_0,2]
```

---

## 💡 核心理解补充

### waiting 和 running 队列

```
一个请求的生命周期:
  新请求 ──→ waiting ──Prefill──→ running ──Decode循环──→ 完成(移出)
                ↑                     │
                │                     │ 显存不够
                └── preempt(踢回) ←───┘
```

三个地方的区别：

```
waiting:        排队等候区（等待被处理）→ 持久
running:        正在服务区（GPU 在处理）→ 持久
scheduled_seqs: 本次 step 的名单（临时的，用完即丢）
```

### Decode 中 popleft 再 extendleft 的原因

```
为什么不直接在 running 里操作？因为循环中间可能踢人：

running = [seq_0, seq_1, seq_2]

popleft seq_0 → running = [seq_1, seq_2]
  → seq_0 需要新 Block → free 空了 → 踢 running.pop() = seq_2
  → running = [seq_1]

popleft seq_1 → running = []
  → 正常处理

scheduled_seqs = [seq_0, seq_1]   ← seq_2 被踢了
extendleft → running = [seq_0, seq_1]

如果不取出来，踢人时可能踢到正在处理的 seq，逻辑会乱
```

---

## 💡 模型代码（qwen3.py + layers/）与 Day 1-2 知识的映射

```
Day 1-2 学的概念              qwen3.py 代码
──────────────────────────────────────────────
Token Embedding               embed_tokens(input_ids)

一层 Transformer Block:        Qwen3DecoderLayer.forward()
  RMSNorm                       input_layernorm（融合 Residual）
  MHA 5 步:                     Qwen3Attention.forward()
    QKV 投影(合并1次GEMM)         qkv_proj → split 拆成 Q,K,V
    分头                          q.view(-1, num_heads, head_dim)
    RoPE                          rotary_emb(positions, q, k)
    Attention                     attn(q, k, v) → flash_attn
    合头 + W_O                    o.flatten → o_proj
  RMSNorm                       post_attention_layernorm
  SwiGLU FFN:                   Qwen3MLP.forward()
    gate_up 合并投影               gate_up_proj（合成一个矩阵，一次 GEMM）
    Swish ⊙ up                    SiluAndMul = F.silu(gate) * up
    down 投影                      down_proj

× N 层                          for layer in self.layers
Final RMSNorm                   self.norm
LM Head                         compute_logits → lm_head
Weight Tying                    lm_head.weight = embed_tokens.weight
```

### RMSNorm 代码和公式的对应

```python
variance = hidden_states.pow(2).mean(-1, keepdim=True)    # RMS² = mean(x²)
hidden_states = hidden_states * torch.rsqrt(variance + eps)  # x / RMS
return self.weight * hidden_states                          # γ × x_norm
```

### RoPE 代码和公式的对应

```python
freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))   # θ_i = base^(-2i/d)
freqs = outer(positions, freqs)                       # angle = position × θ_i
cos, sin = cos(freqs), sin(freqs)
q_embed = q * cos + rotate_half(q) * sin             # 旋转公式
```

### SwiGLU 代码和原理的对应

```python
gate_up = gate_up_proj(x)           # W_gate 和 W_up 合成一个矩阵，一次 GEMM
gate, up = gate_up.chunk(2, dim=-1) # 拆成两半
output = F.silu(gate) * up          # Swish(gate) ⊙ up
output = down_proj(output)          # × W_down
```

### Sampler 的 Gumbel-max 采样

```python
# 不是用 torch.multinomial，而是用 Gumbel-max 技巧
noise = torch.empty_like(probs).exponential_(1)      # 指数分布随机噪声
result = probs / noise.clamp_min_(1e-10)              # 概率除以噪声
sample_tokens = result.argmax(dim=-1)                  # 取最大值

# 数学上等价于按概率采样，但实现上更快（除法+argmax 比 multinomial 快）
```

---

## 📝 注释实践成果

已完成注释的文件：
- [x] scheduler.py（Prefill + Decode 完整注释）
- [x] llm_engine.py（step + generate 注释）

注释原则：用自己的话解释"在做什么"，不是翻译代码本身。

---

## 📊 源码阅读进度

```
engine/:     ✅ 全部完成 + scheduler/llm_engine 已加中文注释
layers/:     ✅ 全部完成
models/:     ✅ 全部完成
utils/:      ✅ 全部完成

1358 / 1358 行 = 100% 📖 源码阅读完成！
```

---

## 🤔 今日反思

今天是最有收获的一天。

上午的回顾测试暴露了一个核心问题——step() 和 schedule() 的职责分不清。通过纠正后明确了：step 是老板（串联三步），schedule 是调度员（选人 + 管 Block），model_runner 是厨房（GPU 干活）。

写 scheduler.py 的注释时，逼自己想清楚了每一行在做什么。特别是 Decode 部分的 popleft + preempt + extendleft 逻辑，一开始觉得很绕，写完注释后才理解为什么要先取出来再放回去——因为中间可能踢人，取出来后 running 里只剩"还没被选中的请求"，踢人只会踢它们。

完整流程例子是今天最大的收获。之前每个函数单独看都能理解，但串起来就模糊。走完 7 步的完整例子后（含 Prefill → Decode → 请求完成 → 显存不够 → 踢人 → 重新 Prefill），整个系统的运作方式终于清晰了。**以后复习看这个例子就够了。**

模型代码部分（qwen3.py + layers）读起来明显轻松，因为都是 Day 1-2 学过的概念。最大的感受是"原来代码长这样"——RMSNorm 就是 pow(2).mean + rsqrt，RoPE 就是 cos/sin + rotate_half，SwiGLU 就是 silu(gate) * up。学过原理后看代码，公式和代码之间的对应关系一目了然。

---

## 🎯 Day 9 计划

1. 继续给 block_manager.py 和 attention.py 写中文注释
2. 尝试本地跑通 nano-vllm（安装依赖 + 下载模型）
3. 如果跑通 → 跑 benchmark 看看实际性能
