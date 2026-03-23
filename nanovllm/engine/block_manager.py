from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        #Prefix Cache命中计数
        self.num_cache_hit = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        # 如果没有分配块才继续执行
        assert not seq.block_table
        h = -1
        cache_miss = False
        #* 遍历这个请求的每个逻辑块
        for i in range(seq.num_blocks):
            #* 取出第i个逻辑块包含的token ids
            token_ids = seq.block(i)
            #* 如果token数等于块的大小，或者说块正好被填满，则算hash，没满的内容还会变，所以不能缓存；注意这里不是查表或者注册
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            #* 查hash表
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            # 如果没有命中缓存，从free list列表里取一个新的块，然后分配
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                #表示已经被缓存的token(又增加了一个块的大小)，这些token不用重复计算
                seq.num_cached_tokens += self.block_size
                #cache命中计数
                self.num_cache_hit += 1
                #* 如果块正在被使用，引用次数加1
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    #* 这个块之前被用过，注册过hash，但被释放了，所以重新激活
                    block = self._allocate_block(block_id)
            #* 如果block是满的，则注册到hash表供后续查找
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        #* 在block里遍历，从后往前遍历是因为后面的block依赖前面的，所以先释放没有依赖的block
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            #注意这里引用次数等于0的时候才完全释放，因为只是减1的话有可能有别的块还在引用这个
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        #清空已缓存的token，释放块
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        #判断能不能加一个块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        #第一种情况：刚好需要分配新的块：
        if len(seq) % self.block_size == 1:
            #*上一个block已满，不等于-1
            assert last_block.hash != -1
            #* 取free list的第一个块，表示要分配到这个，然后标记为已用
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        #第二种情况：正好满足块的大小
        elif len(seq) % self.block_size == 0:
            #* 之前没满，hash=-1
            assert last_block.hash == -1
            #最后一个块
            token_ids = seq.block(seq.num_blocks-1)
            #* 取上一个block的hash值，下面传给compute hash，让当前block的hash包含前缀信息
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            #算hash、存hash
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
