# Day 6 - nano-vllm 源码阅读（二）：Sequence + BlockManager

> 📅 日期：2025年3月18日
> 📍 阶段：阶段 3.5 - nano-vllm 项目
> ⏱️ 学习时长：约 3h

---

## ✅ 今日完成

- [x] `sequence.py` 逐行理解（83行）
- [x] `block_manager.py` 逐行理解（112行）
- [x] Block / Block Table / Free List 在代码中的具体实现
- [x] Prefix Cache 的哈希链式计算机制
- [x] allocate / deallocate / can_append / may_append 四个核心方法
- [x] 引用计数（ref_count）实现共享和释放
- [x] Python 基础补充（self、assert、@property、Enum）

---

## 💡 核心理解

### 1. Sequence —— 一个请求的"身份证"

Sequence 记录了一个请求在系统中的所有状态：

```python
class Sequence:
    seq_id              # 请求编号（自动递增 0, 1, 2...）
    status              # 状态：WAITING / RUNNING / FINISHED
    token_ids           # 所有 token（prompt + 已生成的）
    last_token          # 最后一个 token（Decode 时只需要这个作为输入）
    num_tokens          # 当前总 token 数
    num_prompt_tokens   # prompt 的 token 数（固定不变）
    num_cached_tokens   # Prefix Cache 命中了多少 token
    block_table         # 逻辑块→物理块的映射列表
```

**block_table 的实现**：就是一个普通列表，下标 = 逻辑块编号，值 = 物理块编号。

```
block_table = [3, 0, 7]
  逻辑块 0 → 物理块 3
  逻辑块 1 → 物理块 0
  逻辑块 2 → 物理块 7
```

**block_size = 256**：每个 Block 最多存 256 个 token 的 KV Cache。不是最大 token 数，是每个 Block 的容量。

```
600 个 token, block_size=256:
  需要 ⌈600/256⌉ = 3 个 Block
  Block 0: token 0-255   (满)
  Block 1: token 256-511 (满)
  Block 2: token 512-599 (只用了 88 个位置)
```

**关键属性**：
- `num_completion_tokens = num_tokens - num_prompt_tokens`：已经生成了多少 token
- `num_blocks = ⌈num_tokens / block_size⌉`：需要几个 Block
- `last_block_num_tokens`：最后一个 Block 里有几个 token

**append_token()**：Decode 时把新生成的 token 追加到末尾，num_tokens +1。

### 2. Block —— KV Cache 的存储单元

```python
class Block:
    block_id    # 物理块编号
    ref_count   # 引用计数（几个请求在共享这个 Block）
    hash        # Prefix Cache 用的哈希值（-1 表示无效/Block 没满）
    token_ids   # 这个 Block 存的 token ids（用于 prefix cache 内容匹配）
```

**ref_count 的含义**：

```
ref_count = 0  → 没人用 → 在 Free List 里，可以被分配
ref_count = 1  → 1 个请求在用
ref_count = 2  → 2 个请求共享（prefix cache 场景）
```

**hash = -1 的含义**：Block 还没有有效的 hash，两种情况：
- Block 没满（内容还在变化，不能算 hash）
- Block 刚分配，还没注册

只有 Block 满了（block_size 个 token）才会算 hash 并注册到 hash 表，因为满了之后内容不会再变。

### 3. BlockManager —— PagedAttention 的核心实现

**内部数据结构**：

```python
self.blocks            # Block Pool：所有 Block 对象的列表
self.free_block_ids    # Free List：空闲物理块编号（deque）
self.used_block_ids    # 正在使用的物理块编号（set）
self.hash_to_block_id  # Prefix Cache：hash → 物理块编号（dict）
```

**两个映射不要混淆**：

```
block_table（列表，每个请求各一个）:
  "我的第 i 个逻辑块在哪个物理块？"
  逻辑块 → 物理块

hash_to_block_id（字典，全局共享一个）:
  "这组 token 之前有没有被缓存？"
  用于 Prefix Cache 查找
```

### 4. allocate() —— Prefill 时分配 Block

遍历请求的每个逻辑块，对每个块：

```
① 如果 Block 满了 → 算 hash → 查 hash 表
   ├→ hash 命中且内容匹配 → Prefix Cache 命中！
   │    → 共享物理块（ref_count++）
   │    → num_cached_tokens += block_size（跳过计算）
   └→ 不命中 → 从 Free List 取新 Block

② 如果 Block 没满 → h = -1 → 不查 hash → 直接从 Free List 取新 Block

③ seq.block_table.append(block_id)  → 建立逻辑→物理映射
```

### 5. deallocate() —— 请求完成时释放 Block

```
遍历 block_table 中的每个物理块:
  ref_count -= 1
  如果 ref_count == 0 → 没人用了 → 放回 Free List
  如果 ref_count > 0  → 还有别的请求共享 → 不释放
```

共享场景的例子：

```
seq_0 和 seq_1 共享物理块 0（ref_count=2）
  seq_0 完成 → ref_count: 2→1 → 不释放（seq_1 还在用）
  seq_1 完成 → ref_count: 1→0 → 释放！放回 Free List
```

### 6. can_append() vs may_append() —— Decode 时的 Block 管理

```
can_append = 只检查，不动手（"能不能追加？"）
may_append = 实际执行操作（"去追加吧"）
```

调用顺序在 scheduler.py 中：

```python
while not block_manager.can_append(seq):   # 先检查
    preempt(running.pop())                  # 不够 → 踢人
else:
    block_manager.may_append(seq)           # 够了 → 执行
```

**can_append 的逻辑**：

```
token 数 % block_size == 1 → 需要 1 个新 Block（新 Block 的第一个 token）
其他情况                    → 不需要新 Block → 永远返回 True
```

**may_append 的三种情况**（block_size=4 示例）：

```
情况 A: token%4==1 → 分配新 Block，追加到 block_table
  [t0 t1 t2 t3 | t4]  → block_table 从 [0] 变成 [0, 2]

情况 B: token%4==0 → Block 刚好满了，给它算 hash 注册 prefix cache
  [t0 t1 t2 t3 | t4 t5 t6 t7]  → Block 1 注册 hash

情况 C: 其他 → 什么都不做（Block 没满，继续用）
  [t0 t1 t2 t3 | t4 t5]  → 不操作
```

### 7. Prefix Cache 的哈希链式计算

hash 计算包含前一个 Block 的 hash 作为前缀：

```
Block 0 的 hash = hash([t0, t1, ..., t255])
Block 1 的 hash = hash(Block0的hash + [t256, t257, ..., t511])
Block 2 的 hash = hash(Block1的hash + [t512, t513, ..., t767])
```

为什么要链式？防止这种错误共享：

```
seq_A: [1, 2, 3, 4 | 5, 6, 7, 8]
seq_B: [9, 9, 9, 9 | 5, 6, 7, 8]

Block 1 的 token 内容相同 [5, 6, 7, 8]
但前面内容不同 → KV Cache 不同 → 不能共享！

链式 hash 保证：只有完整前缀一致的 Block 才会被共享
```

---

## 🆚 概念到代码的映射

| Day 3 学的概念 | 代码实现 |
|------|------|
| Block（固定大小的 KV Cache 单元） | `class Block`，block_id + ref_count + hash |
| Block Pool | `self.blocks = [Block(i) for i in range(num_blocks)]` |
| Free List | `self.free_block_ids = deque(range(num_blocks))` |
| Block Table（逻辑→物理映射） | `seq.block_table = [3, 0, 7]`（普通列表） |
| 按需分配 | `allocate()` 中从 Free List 取 Block |
| 释放回收 | `deallocate()` 中 ref_count 降到 0 放回 Free List |
| Copy-on-Write 共享 | `ref_count += 1`（多个请求指向同一个物理块） |
| Prefix Cache | `hash_to_block_id` 字典 + 链式哈希 |

---

## 📝 Python 语法收获

| 语法 | 说明 | 例子 |
|------|------|------|
| `self` | 类方法的第一个参数，指"这个对象自己" | `self.token_ids` = 这个对象的 token_ids |
| `assert` | 断言检查，条件不满足则程序崩溃 | `assert 0 <= i < n` |
| `@property` | 让方法像属性一样访问（不用加括号） | `seq.num_blocks` 实际调用了方法 |
| `Enum` | 枚举类型，给状态命名 | `SequenceStatus.WAITING` |
| `auto()` | 枚举中自动分配值 | `WAITING = auto()` → 1 |
| `count()` | itertools 的无限计数器 | `next(counter)` → 0, 1, 2, 3... |
| `copy()` | 浅拷贝列表 | `self.token_ids = copy(token_ids)` |
| `deque` | 双端队列，左右都能高效增删 | `popleft()`, `append()`, `appendleft()` |
| `set` | 集合，查找/添加/删除都是 O(1) | `self.used_block_ids = set()` |
| `dict.get(key, default)` | 查字典，找不到返回默认值 | `hash_to_block_id.get(h, -1)` |
| 列表切片 | `list[start:end]` 取子列表 | `token_ids[0:256]` |
| `//` | 整数除法（向下取整） | `6 // 4 = 1` |
| `reversed()` | 反向遍历 | `for block_id in reversed(block_table)` |
| 布尔值当数字 | True=1, False=0 | `(len(seq) % 4 == 1)` → 0 或 1 |

---

## 📊 源码阅读进度

```
engine/ 目录（核心引擎）:
  [x] llm_engine.py    (93行)    Day 5 ✅
  [x] scheduler.py     (71行)    Day 5 ✅
  [x] sequence.py      (83行)    Day 6 ✅  ← 今天
  [x] block_manager.py (112行)   Day 6 ✅  ← 今天
  [ ] model_runner.py  (251行)   Day 7 🔜

layers/ 目录（模型组件）:
  [ ] attention.py     (75行)    Day 7 🔜
  [ ] rotary_embedding.py (61行) Day 8
  [ ] layernorm.py     (50行)    Day 8
  [ ] linear.py        (153行)   Day 8
  [ ] activation.py    (14行)    Day 8
  [ ] embed_head.py    (66行)    Day 8
  [ ] sampler.py       (15行)    Day 8

models/ 目录:
  [ ] qwen3.py         (215行)   Day 8

进度: 359 / 1358 行 ≈ 26%（但核心逻辑已覆盖 ~60%）
```

---

## 🤔 今日反思

今天读完了 Sequence 和 BlockManager，这两个文件加起来 195 行，是 PagedAttention 在代码层面的具体实现。

最大的收获是看到了 Day 3 学的概念如何变成代码：Block Table 就是一个普通列表、Free List 就是一个 deque、引用计数就是一个整数字段。之前学原理时觉得很抽象的东西，代码实现其实很直接。

Prefix Cache 的链式哈希设计很巧妙——每个 Block 的 hash 包含前一个 Block 的 hash，保证只有完整前缀一致的 Block 才会被共享。之前在 Day 4 学 SGLang 的 RadixAttention 时觉得前缀缓存很复杂，看到代码才发现核心就是一个 hash 表查找。

can_append 和 may_append 一开始分不清，搞明白后发现就是"先问能不能做，再真正去做"的模式——检查和执行分开，中间可以插入踢人逻辑。

有一个诚实的感受：看完代码觉得"明白了"，但让我自己写肯定写不出来。不过对于目前的目标来说，能读懂和讲清楚就够了。后续可以通过给代码写中文注释来加深理解。

---

## 🎯 Day 7 计划

1. `attention.py`（75行）→ KV Cache 存取 + flash_attn 调用
2. `model_runner.py` 核心部分 → prepare_prefill / prepare_decode / allocate_kv_cache
3. 如果时间够 → CUDA Graph 部分（capture_cudagraph）
