from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        #增加计数器
        self.num_prefill = 0
        self.num_decode = 0
        self.num_preempt = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 判断条件：当waiting队列有请求，进入循环，开始prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            #* 每次循环都看队首的请求，因为每次循环末尾都会移走
            seq = self.waiting[0]
            # 检查：如果序列的长度超出范围，或者没有块能分配，中断
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            #* 不是token数加1，意思是已选的请求加1
            num_seqs += 1
            # 分配块
            self.block_manager.allocate(seq)
            #* len(seq)是这个请求有多少个token，seq.num_cached_tokens是有多少被prefix cache命中
            #* 所以差值就是实际需要gpu计算的token数
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 更改状态为running、从waiting队列里移出，加入running队列
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            #* 这一步要处理的请求(名单加1)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            #prefill计数加1
            self.num_prefill += 1
            return scheduled_seqs, True

        # decode
        #当running队列有请求，并且没有超出最大能处理的范围，进入decode
        while self.running and num_seqs < self.max_num_seqs:
            # 把请求从running队列里取出，表明开始处理
            seq = self.running.popleft()
            # 当分配不了新的块时，开始踢请求
            while not self.block_manager.can_append(seq):
                #* running里还有别的请求
                if self.running:
                    #* 踢别人
                    self.preempt(self.running.pop())
                #* running空了，没人可踢
                else:
                    #* 踢自己
                    self.preempt(seq)
                    break
            # 能分配新的块时，已选的请求+1，然后分配块
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                # 这一步要处理的请求(名单+1)
                scheduled_seqs.append(seq)
        #* 确保至少选到了一个请求
        assert scheduled_seqs
        #decode计数加1
        self.num_decode += 1
        #* 把选好的请求放回running队列头部
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.num_preempt += 1
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def reset_stats(self):
        self.num_prefill = 0
        self.num_decode = 0
        self.num_preempt = 0
        self.block_manager.num_cache_hit = 0