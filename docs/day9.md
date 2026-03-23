# Day 9 - nano-vllm 源码注释（三）：BlockManager 深入 + 完整例子

> 📅 日期：2025年3月21日
> 📍 阶段：阶段 3.5 - nano-vllm 项目
> ⏱️ 学习时长：约 2h

---

## ✅ 今日完成

- [x] `block_manager.py` 的 `allocate()` 逐行注释 + 问题解答
- [x] `block_manager.py` 的 `deallocate()` / `can_append()` / `may_append()` 注释
- [x] 完整例子：allocate → Decode(can/may_append) → deallocate 全流程
- [x] Prefix Cache 共享 + ref_count 释放的完整演示
- [x] 搞清楚 hash 的三步流程：算 hash → 查 hash → 存 hash

---

## 💡 核心理解

### 1. allocate() 的完整逻辑

Prefill 时为请求分配 Block。遍历每个逻辑块，先查 Prefix Cache，命中就共享，没命中就分配新 Block：

```python
def allocate(self, seq):
    h = -1                  # 上一个 Block 的 hash（-1 = 初始，没有前缀）
    cache_miss = False      # 一旦 True 就不会变回 False（链式 hash 决定的）

    for i in range(seq.num_blocks):          # 遍历每个逻辑块
        token_ids = seq.block(i)             # 取这个块包含的 token ids
        
        # ① 算 hash（只有满的 Block 才算）
        h = compute_hash(token_ids, h) if len(token_ids) == block_size else -1
        
        # ② 查 hash（看有没有之前缓存过的）
        block_id = hash_to_block_id.get(h, -1)
        if block_id == -1 or 内容不匹配:
            cache_miss = True

        if cache_miss:
            # 没命中 → 从 Free List 取新 Block
        else:
            # 命中 → 共享（ref_count++）或复用已释放的 Block

        # ③ 存 hash（注册到哈希表，供后续请求查找）
        if h != -1:
            hash_to_block_id[h] = block_id

        seq.block_table.append(block_id)     # 建立逻辑→物理映射
```

**hash 的三步分离**：

```
① 算 hash:  h = compute_hash(token_ids, 前一个Block的hash)  → 得到一个数字
② 查 hash:  hash_to_block_id.get(h, -1)                     → 看有没有命中
③ 存 hash:  hash_to_block_id[h] = block_id                  → 注册供后续查找
```

**h = -1 的两种含义**：
- 初始值：第一个 Block 没有前缀
- Block 没满：内容还会变，不能缓存

**cache_miss 一旦 True 就不会回 False**：因为 hash 是链式的，Block 1 没命中说明前缀不同，后面所有 Block 的 hash 都不可能匹配。

### 2. deallocate() 的逻辑

```
从后往前遍历 block_table:
  ref_count -= 1
  ref_count == 0 → 放回 Free List
  ref_count > 0  → 不释放（还有别的请求共享）
清空 num_cached_tokens 和 block_table
```

### 3. can_append vs may_append

```
can_append: 只检查，不动手
  token % block_size == 1 → 需要新 Block → 检查 Free List 够不够
  其他                     → 不需要新 Block → 永远返回 True

may_append: 实际执行操作
  token % block_size == 1 → 分配新 Block，追加到 block_table
  token % block_size == 0 → Block 满了，算 hash 注册 prefix cache
  其他                     → 什么都不做
```

### 4. 链式 hash 详解

每个 Block 的 hash 包含前一个 Block 的 hash：

```
Block 0 的 hash = hash(Block0的tokens, -1)
Block 1 的 hash = hash(Block1的tokens, Block0的hash)
Block 2 的 hash = hash(Block2的tokens, Block1的hash)  ← 间接包含了 Block 0

所以只需要取"前一个"的 hash，不需要把所有前面的合到一起
因为前一个的 hash 本身就已经包含了更前面所有的信息
```

为什么要链式？防止错误共享：

```
seq_A: [1, 2, 3, 4 | 5, 6, 7, 8]
seq_B: [9, 9, 9, 9 | 5, 6, 7, 8]

Block 1 的 token 内容相同 [5, 6, 7, 8]
但链式 hash 不同（因为 Block 0 不同）
→ 不会错误共享 ✅
```

---

## 🔥 完整例子：BlockManager 全流程

### 设定

```
block_size = 4
Free List: [0, 1, 2, 3, 4, 5]（6 个空闲 Block）
hash_to_block_id = {}

seq_0: token_ids = [10, 20, 30, 40, 50, 60]    6 个 token → 2 个 Block
seq_1: token_ids = [10, 20, 30, 40, 70, 80]    6 个 token → 2 个 Block
                    ^^^^^^^^^^^^^^^^
                    前 4 个 token 和 seq_0 一样！
```

### 第一步：allocate(seq_0)

```
h = -1, cache_miss = False

─── i=0 ───
  token_ids = [10, 20, 30, 40]（4 个 = block_size → 算 hash）
  h = compute_hash([10,20,30,40], -1) → h = 77701

  查哈希表: get(77701, -1) → -1（没有）→ cache_miss = True

  cache_miss → 从 Free List 取 Block 0
  free: [0,1,2,3,4,5] → [1,2,3,4,5]
  Block 0: ref_count=1

  h=77701 != -1 → 注册: hash_to_block_id[77701] = 0
  seq_0.block_table = [0]

─── i=1 ───
  token_ids = [50, 60]（2 个 ≠ block_size → h = -1）

  cache_miss 还是 True → 从 Free List 取 Block 1
  free: [1,2,3,4,5] → [2,3,4,5]
  Block 1: ref_count=1

  h=-1 → 不注册
  seq_0.block_table = [0, 1]

结果:
  seq_0.block_table = [0, 1]
  seq_0.num_cached_tokens = 0
  free = [2, 3, 4, 5]
  hash_to_block_id = {77701: 0}
  Block 0: ref_count=1, hash=77701
  Block 1: ref_count=1, hash=-1
```

### 第二步：allocate(seq_1) —— Prefix Cache 命中！

```
h = -1, cache_miss = False

─── i=0 ───
  token_ids = [10, 20, 30, 40]（和 seq_0 的 Block 0 一样！）
  h = compute_hash([10,20,30,40], -1) → h = 77701（和 seq_0 一样！）

  查哈希表: get(77701, -1) → block_id = 0（找到了！）
  blocks[0].token_ids == [10,20,30,40] → 内容匹配！
  → cache_miss 保持 False → 命中！

  命中:
    seq_1.num_cached_tokens += 4 → 4（前 4 个 token 不用重新计算！）
    block_id=0 in used_block_ids → True（seq_0 在用）
    → 共享！Block 0: ref_count 1→2

  seq_1.block_table = [0]    ← 和 seq_0 指向同一个物理块！

─── i=1 ───
  token_ids = [70, 80]（和 seq_0 不同了）
  h = -1（没满）

  get(-1, -1) → -1 → cache_miss = True

  cache_miss → 从 Free List 取 Block 2
  free: [2,3,4,5] → [3,4,5]
  Block 2: ref_count=1

  seq_1.block_table = [0, 2]

结果:
  seq_0.block_table = [0, 1]
  seq_1.block_table = [0, 2]    ← 物理块 0 被两个请求共享！
  seq_1.num_cached_tokens = 4
  free = [3, 4, 5]
  Block 0: ref_count=2（共享中）
  Block 1: ref_count=1
  Block 2: ref_count=1

  GPU 显存:
  ┌────────┬────────┬────────┬────────┬────────┬────────┐
  │Block 0 │Block 1 │Block 2 │Block 3 │Block 4 │Block 5 │
  │seq_0&1 │ seq_0  │ seq_1  │  空闲  │  空闲  │  空闲  │
  │共享!   │        │        │        │        │        │
  └────────┴────────┴────────┴────────┴────────┴────────┘
```

### 第三步：Decode 循环 —— can_append + may_append

跟踪 seq_0 每一步的变化：

**Decode Step 1：seq_0 生成 token 70，num_tokens 6→7**

```
can_append: 7 % 4 == 3 → 不需要新 Block → True
may_append: 7 % 4 == 3 → else 分支 → 什么都不做

Block 1 还没满，继续往里放。
```

**Decode Step 2：seq_0 生成 token 80，num_tokens 7→8**

```
can_append: 8 % 4 == 0 → 不需要新 Block → True
may_append: 8 % 4 == 0 → elif 分支 → Block 1 刚好满了！

  token_ids = seq_0.block(1) = [50, 60, 70, 80]
  prefix = blocks[block_table[-2]].hash = Block 0 的 hash = 77701
  h = compute_hash([50,60,70,80], 77701) → h = 93205

  Block 1: 更新 hash=93205, token_ids=[50,60,70,80]
  hash_to_block_id[93205] = 1    ← 注册！后续请求可以命中

  hash_to_block_id = {77701: 0, 93205: 1}
```

**Decode Step 3：seq_0 生成 token 90，num_tokens 8→9**

```
can_append: 9 % 4 == 1 → 需要 1 个新 Block
  free = [3, 4, 5] → 有 3 个 ≥ 1 → True

may_append: 9 % 4 == 1 → if 分支 → 分配新 Block！
  assert last_block.hash != -1  ← Block 1 hash=93205 ✅
  block_id = free_block_ids[0] = 3
  _allocate_block(3): free=[4,5], used={0,1,2,3}
  block_table.append(3) → seq_0.block_table = [0, 1, 3]

seq_0 现在有 3 个 Block:
  Block 0: [10,20,30,40]  hash=77701  (和 seq_1 共享)
  Block 1: [50,60,70,80]  hash=93205
  Block 3: [90]           hash=-1（没满）
```

### 第四步：deallocate(seq_0) —— seq_0 完成

```
seq_0.block_table = [0, 1, 3]，从后往前遍历:

  Block 3: ref_count 1→0 → 释放！free=[4,5,3]
  Block 1: ref_count 1→0 → 释放！free=[4,5,3,1]
  Block 0: ref_count 2→1 → 不释放！seq_1 还在用！

seq_0.block_table = []（清空）

结果:
  free = [4, 5, 3, 1]
  used = {0, 2}
  Block 0: ref_count=1（还有 seq_1 在用）
```

### 第五步：deallocate(seq_1) —— seq_1 也完成

```
seq_1.block_table = [0, 2]:

  Block 2: ref_count 1→0 → 释放！free=[4,5,3,1,2]
  Block 0: ref_count 1→0 → 释放！free=[4,5,3,1,2,0]

  全部回收！
```

### 完整时间线

```
操作                   Block 0      Block 1      Block 2      Block 3    free
                       (ref/hash)   (ref/hash)   (ref/hash)   (ref)
─────────────────────────────────────────────────────────────────────────────────
初始                    -            -            -            -         [0,1,2,3,4,5]
allocate(seq_0)        1/77701      1/-1         -            -         [2,3,4,5]
allocate(seq_1)        2/77701      1/-1         1/-1         -         [3,4,5]
                       ^共享！       
Decode: 7tok           (不变)       (不变)       (不变)       -         [3,4,5]
Decode: 8tok           (不变)       1/93205      (不变)       -         [3,4,5]
                                    ^满了→注册hash
Decode: 9tok           (不变)       (不变)       (不变)       1/-1      [4,5]
                                                              ^新分配
deallocate(seq_0)      1/77701      释放         (不变)       释放      [4,5,3,1]
                       ^2→1（不释放）
deallocate(seq_1)      释放         -            释放         -         [4,5,3,1,2,0]
                       ^1→0（释放）                                     全部回收！
```

---

## 📝 注释实践成果

已完成注释的文件：
- [x] scheduler.py 完整注释（Day 8）
- [x] llm_engine.py step() + generate() 注释（Day 8）
- [x] block_manager.py allocate / deallocate / can_append / may_append 注释（Day 9）

待完成：
- [ ] attention.py 注释

---

## 🤔 关键问题澄清

### seq.block(i) 是什么

不是"把块分配给 token"，而是**取出第 i 个逻辑块包含的 token ids**：

```
token_ids = [10, 20, 30, 40, 50, 60], block_size = 4
seq.block(0) = [10, 20, 30, 40]    ← 前 4 个
seq.block(1) = [50, 60]            ← 后 2 个（没满）
```

### num_batched_tokens 是什么

```python
num_batched_tokens += len(seq) - seq.num_cached_tokens
```

累计实际需要 GPU 计算的 token 数。减去 prefix cache 命中的部分（已缓存的不用重新算），防止一步处理太多 token 导致显存不够。

### allocate 中 Prefix Cache 命中的两种子情况

```
命中后:
  if block_id in used_block_ids:
    → Block 正在被别的请求使用 → 共享（ref_count++）
  else:
    → Block 之前用过但已释放（在 Free List 里）→ 重新激活
```

---

## 📊 项目进度

```
源码阅读:     ✅ 100% 完成
中文注释:     scheduler.py ✅ + llm_engine.py ✅ + block_manager.py ✅
              还剩 attention.py
本地跑通:     还没开始
动手实践:     还没确定方向

剩余计划:
  Day 10:  attention.py 注释 + 本地跑通
  Day 11:  动手实践（写注释文档 / 简化复现 / 加功能）
  Day 12:  整理成果 + 推 GitHub + 更新简历
```

---

## 💭 今日反思

今天最大的收获是通过完整例子把 BlockManager 的四个核心方法全部串起来了。之前单独看每个函数能理解，但不知道它们什么时候被调用、按什么顺序执行。走完一个从 allocate 到 Decode 到 deallocate 的完整生命周期后，整个 PagedAttention 的运作方式变得非常清晰。

allocate 的 Prefix Cache 逻辑一开始比较绕——hash 的"算→查→存"三步是分开的，代码里混在一起容易搞混。搞清楚之后发现核心就是：算出 hash → 查有没有人注册过 → 命中就共享/复用，没命中就分配新 Block → 最后把自己的 hash 注册上去。

一个关键理解：hash 的三步分离。不是"计算 hash 然后存入哈希表"这一个动作，而是 ① 算出一个数字 ② 用这个数字查表 ③ 把这个数字存入表。三步各有不同的目的。

may_append 的三种情况通过例子也清楚了：
- token%4==1：刚进入新 Block → 分配
- token%4==0：Block 满了 → 算 hash 注册 prefix cache
- 其他：Block 没满 → 什么都不做
